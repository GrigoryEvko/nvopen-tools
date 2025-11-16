// Function: sub_1D1FC50
// Address: 0x1d1fc50
//
__int64 __fastcall sub_1D1FC50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  int v5; // eax
  _QWORD *v6; // rcx
  int v7; // eax
  int v8; // edx
  __int64 v9; // rax
  _QWORD *v10; // rcx
  __int64 result; // rax
  unsigned int v12; // edx
  __int64 v13; // rax
  unsigned int v14; // eax
  unsigned int v15; // r14d
  __int64 v16; // rax
  unsigned int v17; // r14d
  unsigned int v20; // esi
  int v21; // eax
  int v22; // eax
  unsigned int v23; // [rsp+Ch] [rbp-64h]
  unsigned int v24; // [rsp+Ch] [rbp-64h]
  __int64 *v25; // [rsp+10h] [rbp-60h] BYREF
  __int64 v26; // [rsp+18h] [rbp-58h] BYREF
  __int64 v27; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v30; // [rsp+38h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 16);
  v26 = 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64 **, __int64 *))(*(_QWORD *)v4 + 1104LL))(
          v4,
          a2,
          &v25,
          &v26) )
    goto LABEL_2;
  v13 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
  v14 = sub_15A95F0(v13, *v25);
  v28 = v14;
  v15 = v14;
  if ( v14 <= 0x40 )
  {
    v27 = 0;
    v30 = v14;
    v29 = 0;
  }
  else
  {
    sub_16A4EF0((__int64)&v27, 0, 0);
    v30 = v15;
    sub_16A4EF0((__int64)&v29, 0, 0);
  }
  v16 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
  sub_14BB090((__int64)v25, (__int64)&v27, v16, 0, 0, 0, 0, 0);
  v17 = v28;
  if ( v28 > 0x40 )
  {
    v21 = sub_16A58F0((__int64)&v27);
    v20 = v30;
    LODWORD(_RCX) = v21;
    if ( !v21 )
    {
LABEL_19:
      if ( v20 > 0x40 && v29 )
      {
        j_j___libc_free_0_0(v29);
        v17 = v28;
      }
      if ( v17 > 0x40 && v27 )
        j_j___libc_free_0_0(v27);
LABEL_2:
      v5 = *(unsigned __int16 *)(a2 + 24);
      if ( v5 == 14 || v5 == 36 )
      {
        v8 = *(_DWORD *)(a2 + 84);
        LODWORD(v10) = 0;
      }
      else
      {
        if ( !sub_1D1FBF0(a1, a2) )
          return 0;
        v6 = *(_QWORD **)(a2 + 32);
        v7 = *(unsigned __int16 *)(*v6 + 24LL);
        if ( v7 != 36 && v7 != 14 )
          return 0;
        v8 = *(_DWORD *)(*v6 + 84LL);
        v9 = *(_QWORD *)(v6[5] + 88LL);
        v10 = *(_QWORD **)(v9 + 24);
        if ( *(_DWORD *)(v9 + 32) > 0x40u )
          v10 = (_QWORD *)*v10;
      }
      if ( v8 != 0x80000000 )
      {
        v12 = (unsigned int)v10
            | *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 8LL)
                        + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 56LL) + 32LL) + v8)
                        + 16);
        return v12 & -v12;
      }
      return 0;
    }
LABEL_27:
    v22 = 0x80000000;
    if ( (unsigned int)_RCX <= 0x1E )
      v22 = 1 << _RCX;
    result = ((unsigned int)v26 | v22) & -((unsigned int)v26 | v22);
    if ( v20 <= 0x40 )
      goto LABEL_30;
    goto LABEL_34;
  }
  _RAX = ~v27;
  __asm { tzcnt   rcx, rax }
  if ( v27 != -1 )
  {
    v20 = v30;
    if ( !(_DWORD)_RCX )
      goto LABEL_19;
    goto LABEL_27;
  }
  result = ((unsigned int)v26 | 0x80000000) & -((unsigned int)v26 | 0x80000000);
  if ( v30 <= 0x40 )
    return result;
LABEL_34:
  if ( v29 )
  {
    v24 = result;
    j_j___libc_free_0_0(v29);
    v17 = v28;
    result = v24;
  }
LABEL_30:
  if ( v17 > 0x40 && v27 )
  {
    v23 = result;
    j_j___libc_free_0_0(v27);
    return v23;
  }
  return result;
}

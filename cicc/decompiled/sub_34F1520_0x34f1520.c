// Function: sub_34F1520
// Address: 0x34f1520
//
__int64 __fastcall sub_34F1520(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned int v3; // r12d
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rdi
  __int64 (*v14)(); // rax
  __int64 v15; // rdx
  unsigned int v16; // eax
  unsigned int v17; // r12d
  unsigned __int64 v18; // rsi
  bool v19; // zf
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // rax
  __int64 *v22; // r14
  unsigned int v23; // r13d
  __int64 *v24; // rdx
  __int64 v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int8 v27; // [rsp+7h] [rbp-99h]
  bool v28; // [rsp+17h] [rbp-89h] BYREF
  __int64 v29; // [rsp+18h] [rbp-88h] BYREF
  unsigned __int64 v30; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v31; // [rsp+28h] [rbp-78h]
  unsigned __int64 v32; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-68h]
  unsigned __int64 v34; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+48h] [rbp-58h]
  unsigned __int64 v36; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+58h] [rbp-48h]
  unsigned __int64 v38; // [rsp+60h] [rbp-40h] BYREF
  __int64 v39; // [rsp+68h] [rbp-38h]

  if ( !a2 )
    return 0;
  v7 = **(_QWORD **)a1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v7 )
    BUG();
  v8 = *(_QWORD *)v7;
  if ( (*(_QWORD *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v26 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = v26;
      if ( (*(_BYTE *)(v26 + 44) & 4) == 0 )
        break;
      v8 = *(_QWORD *)v26;
    }
  }
  if ( v7 == *(_QWORD *)(*(_QWORD *)a1 + 24LL) + 48LL )
    return 0;
  while ( (unsigned int)sub_2E8E710(v7, a2, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL), 0, 1) == -1 )
  {
    v9 = (_QWORD *)(*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL);
    v10 = v9;
    if ( !v9 )
      BUG();
    v7 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *v9;
    if ( (v11 & 4) == 0 && (*((_BYTE *)v10 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v12 = v11 & 0xFFFFFFFFFFFFFFF8LL;
        v7 = v12;
        if ( (*(_BYTE *)(v12 + 44) & 4) == 0 )
          break;
        v11 = *(_QWORD *)v12;
      }
    }
    if ( v7 == *(_QWORD *)(*(_QWORD *)a1 + 24LL) + 48LL )
      return 0;
  }
  if ( !v7 )
    return 0;
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 200LL);
  v14 = *(__int64 (**)())(*(_QWORD *)v13 + 552LL);
  if ( v14 == sub_2F28CE0 )
    return 0;
  v27 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64 *))v14)(v13, v7, a2, &v29);
  if ( !v27 )
    return 0;
  v38 = sub_2FF6F50(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 208LL), a2, *(_QWORD *)(a1 + 16));
  v39 = v15;
  v16 = sub_CA1930(&v38);
  v31 = v16;
  v17 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43690((__int64)&v30, v29, 1);
    v33 = v17;
    sub_C43690((__int64)&v32, a3, 0);
  }
  else
  {
    v32 = a3;
    v33 = v16;
    v18 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v16) & v29;
    v19 = v16 == 0;
    v20 = 0;
    if ( !v19 )
      v20 = v18;
    v30 = v20;
  }
  sub_C4A7C0((__int64)&v34, (__int64)&v30, (__int64)&v32, &v28);
  v3 = v28;
  if ( v28 )
  {
    v3 = 0;
  }
  else
  {
    v21 = **(_QWORD **)(a1 + 24);
    v37 = 64;
    v36 = v21;
    sub_C45F70((__int64)&v38, (__int64)&v34, (__int64)&v36, &v28);
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    v22 = (__int64 *)v38;
    v23 = v39;
    v36 = v38;
    v37 = v39;
    if ( !v28 )
    {
      if ( (unsigned int)v39 <= 0x40 )
      {
        v24 = *(__int64 **)(a1 + 24);
        v25 = 0;
        if ( (_DWORD)v39 )
          v25 = (__int64)(v38 << (64 - (unsigned __int8)v39)) >> (64 - (unsigned __int8)v39);
      }
      else
      {
        if ( v23 - (unsigned int)sub_C444A0((__int64)&v36) > 0x40 )
        {
          v3 = 0;
LABEL_33:
          if ( v36 )
            j_j___libc_free_0_0(v36);
          goto LABEL_41;
        }
        v24 = *(__int64 **)(a1 + 24);
        v25 = *v22;
      }
      *v24 = v25;
      v3 = v27;
      v23 = v37;
    }
    if ( v23 > 0x40 )
      goto LABEL_33;
  }
LABEL_41:
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  return v3;
}

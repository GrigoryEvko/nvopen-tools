// Function: sub_F8C070
// Address: 0xf8c070
//
__int64 __fastcall sub_F8C070(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  int v6; // eax
  __int64 *v7; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // esi
  _BYTE v19[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v20; // [rsp+20h] [rbp-70h]
  _BYTE v21[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v5 = sub_D9ABD0(a2);
  v6 = *(_DWORD *)(a2 + 48);
  if ( (v6 & 1) == 0 )
  {
    if ( (v6 & 2) == 0 )
    {
LABEL_3:
      v7 = (__int64 *)sub_BD5C60(a3);
      return sub_ACD720(v7);
    }
    v11 = sub_F8AEF0((__int64 **)a1, v5, a3, 1);
LABEL_14:
    if ( !v11 )
      goto LABEL_3;
    return v11;
  }
  v9 = sub_F8AEF0((__int64 **)a1, v5, a3, 0);
  v10 = v9;
  if ( (*(_BYTE *)(a2 + 48) & 2) != 0 )
  {
    v12 = sub_F8AEF0((__int64 **)a1, v5, a3, 1);
    v13 = v12;
    if ( v10 )
    {
      if ( !v12 )
        return v10;
      v14 = *(_QWORD *)(a1 + 600);
      v20 = 257;
      v11 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v14 + 16LL))(v14, 29, v10, v12);
      if ( !v11 )
      {
        v22 = 257;
        v11 = sub_B504D0(29, v10, v13, (__int64)v21, 0, 0);
        (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 608) + 16LL))(
          *(_QWORD *)(a1 + 608),
          v11,
          v19,
          *(_QWORD *)(a1 + 576),
          *(_QWORD *)(a1 + 584));
        v15 = *(_QWORD *)(a1 + 520);
        v16 = v15 + 16LL * *(unsigned int *)(a1 + 528);
        while ( v16 != v15 )
        {
          v17 = *(_QWORD *)(v15 + 8);
          v18 = *(_DWORD *)v15;
          v15 += 16;
          sub_B99FD0(v11, v18, v17);
        }
      }
      return v11;
    }
    v11 = v12;
    goto LABEL_14;
  }
  v11 = v9;
  if ( !v9 )
    goto LABEL_3;
  return v11;
}

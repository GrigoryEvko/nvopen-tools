// Function: sub_EF4590
// Address: 0xef4590
//
__int64 __fastcall sub_EF4590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdx
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r13
  char v13; // al
  __int64 *v14; // rsi
  __int64 *v15; // rax
  __int64 *v16; // r12
  __int64 *v17; // rax
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rax
  char v28; // [rsp+7h] [rbp-D9h]
  __int64 *v29; // [rsp+18h] [rbp-C8h] BYREF
  __int64 v30[2]; // [rsp+20h] [rbp-C0h] BYREF
  _QWORD v31[22]; // [rsp+30h] [rbp-B0h] BYREF

  v6 = *(_QWORD *)(a1 + 8);
  v7 = *(char **)a1;
  if ( (unsigned __int64)(v6 - *(_QWORD *)a1) > 1 && *(_WORD *)v7 == 28260 )
  {
    *(_QWORD *)a1 = v7 + 2;
    if ( (char *)v6 == v7 + 2 || (unsigned int)(v7[2] - 48) > 9 )
      v12 = sub_EED260((char **)a1, a2, v6, (__int64)(v7 + 2), a5, a6);
    else
      v12 = sub_EF0590((__int64 *)a1);
    if ( v12 )
    {
      v13 = *(_BYTE *)(a1 + 937);
      v30[0] = (__int64)v31;
      v28 = v13;
      v30[1] = 0x2000000002LL;
      v31[0] = 50;
      sub_D953B0((__int64)v30, v12, v8, v9, v10, v11);
      v14 = v30;
      v15 = sub_C65B40(a1 + 904, (__int64)v30, (__int64 *)&v29, (__int64)off_497B2F0);
      v16 = v15;
      if ( v15 )
      {
        v16 = v15 + 1;
        if ( (_QWORD *)v30[0] != v31 )
          _libc_free(v30[0], v30);
        v30[0] = (__int64)v16;
        v17 = sub_EE6840(a1 + 944, v30);
        if ( v17 )
        {
          v18 = (__int64 *)v17[1];
          if ( v18 )
            v16 = v18;
        }
        if ( *(__int64 **)(a1 + 928) == v16 )
          *(_BYTE *)(a1 + 936) = 1;
      }
      else
      {
        if ( v28 )
        {
          v27 = sub_CD1D40((__int64 *)(a1 + 808), 32, 3);
          *(_QWORD *)v27 = 0;
          v14 = (__int64 *)v27;
          v16 = (__int64 *)(v27 + 8);
          *(_WORD *)(v27 + 16) = 16434;
          LOBYTE(v27) = *(_BYTE *)(v27 + 18);
          v14[3] = v12;
          *((_BYTE *)v14 + 18) = v27 & 0xF0 | 5;
          v14[1] = (__int64)&unk_49E0088;
          sub_C657C0((__int64 *)(a1 + 904), v14, v29, (__int64)off_497B2F0);
        }
        if ( (_QWORD *)v30[0] != v31 )
          _libc_free(v30[0], v14);
        *(_QWORD *)(a1 + 920) = v16;
      }
      return (__int64)v16;
    }
    return 0;
  }
  sub_EE3B50((const void **)a1, 2u, "on");
  v29 = (__int64 *)sub_EF3FC0(a1, 0);
  v16 = v29;
  if ( !v29 )
    return 0;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 && **(_BYTE **)a1 == 73 )
  {
    v30[0] = sub_EEFA10(a1, 0, v19, v20, v21, v22);
    v16 = (__int64 *)v30[0];
    if ( v30[0] )
      return sub_EE7CC0(a1 + 808, (__int64 *)&v29, (unsigned __int64 *)v30, v24, v25, v26);
  }
  return (__int64)v16;
}

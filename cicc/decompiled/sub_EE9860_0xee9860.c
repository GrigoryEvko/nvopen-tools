// Function: sub_EE9860
// Address: 0xee9860
//
unsigned __int64 __fastcall sub_EE9860(__int64 a1, unsigned __int64 a2)
{
  _BYTE *v3; // rax
  __int64 v5; // rdx
  unsigned __int64 v6; // r14
  char v7; // al
  _QWORD *v8; // rax
  __int64 *v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rax
  __int16 v12; // dx
  __int64 v13; // rsi
  unsigned __int64 v14; // r15
  __int16 v15; // cx
  __int64 *v16; // rdx
  char v17; // [rsp+7h] [rbp-E9h]
  __int64 v18; // [rsp+10h] [rbp-E0h]
  __int64 *v19; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v20[2]; // [rsp+30h] [rbp-C0h] BYREF
  _BYTE v21[176]; // [rsp+40h] [rbp-B0h] BYREF

  v3 = *(_BYTE **)a1;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)a1 )
  {
    while ( *v3 == 66 )
    {
      *(_QWORD *)a1 = v3 + 1;
      v6 = sub_EE3BB0(a1);
      if ( !v6 )
        return 0;
      v7 = *(_BYTE *)(a1 + 937);
      v18 = v5;
      v20[0] = (__int64)v21;
      v17 = v7;
      v20[1] = 0x2000000000LL;
      sub_EE3E30((__int64)v20, 9u, a2, v6, v5, (__int64)v21);
      v8 = sub_C65B40(a1 + 904, (__int64)v20, (__int64 *)&v19, (__int64)off_497B2F0);
      if ( v8 )
      {
        a2 = (unsigned __int64)(v8 + 1);
        if ( (_BYTE *)v20[0] != v21 )
          _libc_free(v20[0], v20);
        v20[0] = a2;
        v9 = sub_EE6840(a1 + 944, v20);
        if ( v9 )
        {
          v10 = v9[1];
          if ( v10 )
            a2 = v10;
        }
        if ( *(_QWORD *)(a1 + 928) == a2 )
          *(_BYTE *)(a1 + 936) = 1;
      }
      else
      {
        if ( !v17 )
        {
          if ( (_BYTE *)v20[0] != v21 )
            _libc_free(v20[0], v20);
          *(_QWORD *)(a1 + 920) = 0;
          return 0;
        }
        v11 = sub_CD1D40((__int64 *)(a1 + 808), 48, 3);
        v12 = *(_WORD *)(v11 + 16);
        *(_QWORD *)v11 = 0;
        v13 = v11;
        v14 = v11 + 8;
        v15 = *(_WORD *)(a2 + 9) & 0xFC0;
        *(_WORD *)(v11 + 16) = v12 & 0xC000 | 9;
        LOWORD(v11) = *(_WORD *)(v11 + 17);
        *(_QWORD *)(v13 + 24) = a2;
        *(_QWORD *)(v13 + 32) = v6;
        *(_QWORD *)(v13 + 40) = v18;
        v16 = v19;
        *(_WORD *)(v13 + 17) = v15 | v11 & 0xF03F;
        *(_QWORD *)(v13 + 8) = &unk_49DF128;
        sub_C657C0((__int64 *)(a1 + 904), (__int64 *)v13, v16, (__int64)off_497B2F0);
        if ( (_BYTE *)v20[0] != v21 )
          _libc_free(v20[0], v13);
        *(_QWORD *)(a1 + 920) = v14;
        a2 = v14;
      }
      v3 = *(_BYTE **)a1;
      if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 8) )
        return a2;
    }
  }
  return a2;
}

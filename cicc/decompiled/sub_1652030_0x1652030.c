// Function: sub_1652030
// Address: 0x1652030
//
void __fastcall sub_1652030(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  const char *v3; // rax
  __int64 v4; // r14
  _BYTE *v5; // rax
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  unsigned __int8 v9; // al
  const char *v10; // rax
  __int64 v11; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v12; // [rsp+8h] [rbp-68h]
  _QWORD *v13; // [rsp+10h] [rbp-60h]
  __int64 v14; // [rsp+18h] [rbp-58h]
  int v15; // [rsp+20h] [rbp-50h]
  _QWORD v16[9]; // [rsp+28h] [rbp-48h] BYREF

  v2 = *(_BYTE *)(a2 + 32) & 0xF;
  if ( (unsigned __int8)v2 <= 8u && (v7 = 445, _bittest64(&v7, v2)) )
  {
    v8 = *(_QWORD *)(a2 - 24);
    if ( !v8 )
    {
      BYTE1(v13) = 1;
      v10 = "Aliasee cannot be NULL!";
LABEL_21:
      v11 = (__int64)v10;
      LOBYTE(v13) = 3;
      sub_164FF40((__int64 *)a1, (__int64)&v11);
      if ( !*(_QWORD *)a1 )
        return;
LABEL_7:
      sub_164FA80((__int64 *)a1, a2);
      return;
    }
    if ( *(_QWORD *)v8 == *(_QWORD *)a2 )
    {
      v9 = *(_BYTE *)(v8 + 16);
      if ( v9 == 5 || v9 <= 3u )
      {
        v12 = v16;
        v13 = v16;
        v14 = 0x100000004LL;
        v15 = 0;
        v16[0] = a2;
        v11 = 1;
        sub_1651AA0(a1, (__int64)&v11, a2, v8);
        if ( v13 != v12 )
          _libc_free((unsigned __int64)v13);
        sub_1651CA0((__int64 *)a1, a2);
        return;
      }
      BYTE1(v13) = 1;
      v10 = "Aliasee should be either GlobalValue or ConstantExpr";
      goto LABEL_21;
    }
    BYTE1(v13) = 1;
    v3 = "Alias and aliasee types should match!";
  }
  else
  {
    BYTE1(v13) = 1;
    v3 = "Alias should have private, internal, linkonce, weak, linkonce_odr, weak_odr, or external linkage!";
  }
  v4 = *(_QWORD *)a1;
  v11 = (__int64)v3;
  LOBYTE(v13) = 3;
  if ( !v4 )
  {
    *(_BYTE *)(a1 + 72) = 1;
    return;
  }
  sub_16E2CE0(&v11, v4);
  v5 = *(_BYTE **)(v4 + 24);
  if ( (unsigned __int64)v5 >= *(_QWORD *)(v4 + 16) )
  {
    sub_16E7DE0(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 24) = v5 + 1;
    *v5 = 10;
  }
  v6 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 72) = 1;
  if ( v6 )
    goto LABEL_7;
}

// Function: sub_155D4D0
// Address: 0x155d4d0
//
__int64 __fastcall sub_155D4D0(__int64 a1, __int64 a2, const char *a3)
{
  size_t v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rcx
  unsigned __int64 v7; // rcx
  char *v8; // r9
  __int64 v9; // rcx
  __int64 v10; // rcx
  unsigned __int64 v12; // rcx
  char *v13; // r9
  __int64 v14; // rcx
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  char v17; // [rsp+14h] [rbp-4Ch] BYREF
  _BYTE v18[11]; // [rsp+15h] [rbp-4Bh] BYREF
  _QWORD *v19; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h]
  _QWORD v21[6]; // [rsp+30h] [rbp-30h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v4 = strlen(a3);
  if ( v4 > 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_22;
  sub_2241490(a1, a3, v4, v5);
  if ( !**(_BYTE **)a2 )
  {
    if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
    {
      sub_2241490(a1, "(", 1, v6);
      v7 = sub_155D4B0(*(__int64 **)(a2 + 8));
      if ( v7 )
      {
        v8 = v18;
        do
        {
          *--v8 = v7 % 0xA + 48;
          v15 = v7;
          v7 /= 0xAu;
        }
        while ( v15 > 9 );
      }
      else
      {
        v17 = 48;
        v8 = &v17;
      }
      v19 = v21;
      sub_155CB60((__int64 *)&v19, v8, (__int64)v18);
      sub_2241490(a1, v19, v20, v9);
      if ( v19 != v21 )
        j_j___libc_free_0(v19, v21[0] + 1LL);
      if ( *(_QWORD *)(a1 + 8) != 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, ")", 1, v10);
        return a1;
      }
    }
LABEL_22:
    sub_4262D8((__int64)"basic_string::append");
  }
  if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_22;
  sub_2241490(a1, "=", 1, v6);
  v12 = sub_155D4B0(*(__int64 **)(a2 + 8));
  if ( v12 )
  {
    v13 = v18;
    do
    {
      *--v13 = v12 % 0xA + 48;
      v16 = v12;
      v12 /= 0xAu;
    }
    while ( v16 > 9 );
  }
  else
  {
    v17 = 48;
    v13 = &v17;
  }
  v19 = v21;
  sub_155CB60((__int64 *)&v19, v13, (__int64)v18);
  sub_2241490(a1, v19, v20, v14);
  if ( v19 == v21 )
    return a1;
  j_j___libc_free_0(v19, v21[0] + 1LL);
  return a1;
}

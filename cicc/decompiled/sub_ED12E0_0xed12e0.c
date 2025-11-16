// Function: sub_ED12E0
// Address: 0xed12e0
//
__int64 __fastcall sub_ED12E0(__int64 a1, int a2, int a3, unsigned __int8 a4)
{
  __int64 v5; // r15
  const char *v6; // r13
  size_t v7; // rdx
  __int64 v8; // rcx
  const char *v10; // r14
  size_t v11; // rax
  const char *v12; // r15
  size_t v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rcx
  bool v16; // [rsp+Fh] [rbp-31h]

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v16 = a3 == 5;
  if ( a3 == 5 && a4 )
  {
    v5 = a2;
    v10 = off_497AD80[a2];
    v11 = strlen(v10);
    sub_2241130(a1, 0, 0, v10, v11);
  }
  else
  {
    v5 = a2;
    if ( a3 == 1 )
    {
      v6 = off_497AE00[a2];
      v7 = strlen(v6);
      if ( v7 <= 0x3FFFFFFFFFFFFFFFLL )
      {
        sub_2241490(a1, v6, v7, v8);
        return a1;
      }
LABEL_13:
      sub_4262D8((__int64)"basic_string::append");
    }
  }
  v12 = off_497AE80[v5];
  v13 = strlen(v12);
  if ( v13 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8) )
    goto LABEL_13;
  sub_2241490(a1, v12, v13, v14);
  if ( (a4 & (a2 == 0)) != 0 && v16 )
  {
    v15 = 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a1 + 8);
    if ( v15 > 0x14 )
    {
      sub_2241490(a1, ",regular,live_support", 21, v15);
      return a1;
    }
    goto LABEL_13;
  }
  return a1;
}

// Function: sub_E0CD60
// Address: 0xe0cd60
//
__int64 __fastcall sub_E0CD60(unsigned __int64 a1, char *a2, __int64 a3, char a4, unsigned __int8 a5)
{
  char *v5; // r14
  unsigned __int64 v7; // r12
  char v9; // dl
  __int64 v10; // rax
  const char *v11; // r12
  size_t v12; // rdx
  __int64 v13; // rcx

  v5 = a2;
  v7 = a1;
  if ( a4 )
  {
    if ( !a1 )
      return 0;
    v9 = *a2;
    if ( *a2 != 46 )
      goto LABEL_4;
    v5 = a2 + 1;
    v7 = a1 - 1;
    sub_2241130(a3, 0, *(_QWORD *)(a3 + 8), ".", 1);
  }
  if ( !v7 )
    return 0;
  v9 = *v5;
LABEL_4:
  v10 = 0;
  while ( v9 == 95 )
  {
    if ( v7 <= ++v10 )
      goto LABEL_10;
    v9 = v5[v10];
  }
  if ( (unsigned __int64)(v10 - 1) <= 3 && v9 == 90 )
  {
    v11 = (const char *)sub_E1D0B0(v7, v5, a5);
    goto LABEL_13;
  }
LABEL_10:
  if ( v7 == 1 )
    return 0;
  if ( *(_WORD *)v5 == 21087 )
  {
    v11 = (const char *)sub_E337A0(v7, v5);
  }
  else
  {
    if ( *(_WORD *)v5 != 17503 )
      return 0;
    v11 = (const char *)sub_E0D940(v7, v5);
  }
LABEL_13:
  if ( !v11 )
    return 0;
  v12 = strlen(v11);
  if ( v12 > 0x3FFFFFFFFFFFFFFFLL - *(_QWORD *)(a3 + 8) )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(a3, v11, v12, v13);
  _libc_free(v11, v11);
  return 1;
}

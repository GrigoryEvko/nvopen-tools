// Function: sub_E219C0
// Address: 0xe219c0
//
__int64 __fastcall sub_E219C0(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 v2; // rcx
  char *v4; // rdi
  char v5; // al
  int v6; // edx
  __int64 v7; // rdx
  unsigned __int64 v8; // r10
  unsigned __int8 v9; // al
  unsigned __int64 v10; // r10

  v2 = *a2;
  if ( !*a2 )
    goto LABEL_9;
  v4 = (char *)a2[1];
  v5 = *v4;
  if ( *v4 == 63 )
  {
    --v2;
    a2[1] = (unsigned __int64)(v4 + 1);
    *a2 = v2;
    if ( v2 )
    {
      v5 = *++v4;
      v6 = v5;
      if ( (unsigned int)(v5 - 48) > 9 )
        goto LABEL_4;
      goto LABEL_10;
    }
LABEL_9:
    *(_BYTE *)(a1 + 8) = 1;
    return 0;
  }
  v6 = v5;
  if ( (unsigned int)(v5 - 48) > 9 )
  {
LABEL_4:
    v7 = 0;
    v8 = 0;
    if ( v5 == 64 )
    {
LABEL_8:
      v10 = v8 + 1;
      a2[1] = (unsigned __int64)&v4[v10];
      *a2 = v2 - v10;
      return v7;
    }
    while ( 1 )
    {
      v9 = v5 - 65;
      if ( v9 > 0xFu )
        goto LABEL_9;
      ++v8;
      v7 = (char)v9 + 16 * v7;
      if ( v8 >= v2 )
        goto LABEL_9;
      v5 = v4[v8];
      if ( v5 == 64 )
        goto LABEL_8;
    }
  }
LABEL_10:
  a2[1] = (unsigned __int64)(v4 + 1);
  *a2 = v2 - 1;
  return v6 - 47;
}

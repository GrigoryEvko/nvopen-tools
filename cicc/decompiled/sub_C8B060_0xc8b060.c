// Function: sub_C8B060
// Address: 0xc8b060
//
unsigned __int64 __fastcall sub_C8B060(__int64 a1, char *a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // r12
  char *v4; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v7; // r15
  char *v8; // r12
  char v9; // si
  char *v10; // r15
  __int64 i; // rax
  unsigned __int64 v12; // rax
  char *j; // r12
  char v14; // si

  v3 = a3;
  v4 = a2;
  result = *(unsigned __int8 *)(a1 + 88);
  *(_DWORD *)(a1 + 84) += a3;
  if ( (_BYTE)result )
  {
    v7 = 64 - (int)result;
    if ( v7 > a3 )
      v7 = a3;
    if ( v7 )
    {
      v8 = a2;
      v4 = &a2[v7];
      do
      {
        v9 = *v8++;
        result = sub_C8B020(a1, v9);
      }
      while ( v8 != v4 );
    }
    v3 = a3 - v7;
  }
  if ( v3 > 0x3F )
  {
    v10 = v4;
    do
    {
      for ( i = 0; i != 64; i += 4 )
        *(_DWORD *)(a1 + i) = _byteswap_ulong(*(_DWORD *)&v10[i]);
      v10 += 64;
      sub_C89F40((_DWORD *)a1);
    }
    while ( (unsigned __int64)(&v4[v3] - v10) > 0x3F );
    v12 = v3 - 64;
    v3 &= 0x3Fu;
    result = v12 & 0xFFFFFFFFFFFFFFC0LL;
    v4 += result + 64;
  }
  for ( j = &v4[v3]; j != v4; result = sub_C8B020(a1, v14) )
    v14 = *v4++;
  return result;
}

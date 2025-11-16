// Function: sub_ED1E60
// Address: 0xed1e60
//
void __fastcall sub_ED1E60(unsigned int *a1, int a2, int a3)
{
  unsigned int v3; // r8d
  unsigned int v4; // eax
  unsigned int v5; // eax
  unsigned __int8 *v6; // rax
  int v7; // edx
  int v8; // ecx
  __int64 v9; // r8
  unsigned __int64 *v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // rdx

  if ( a2 == a3 )
    return;
  v3 = a1[1];
  v4 = v3;
  if ( a2 == 1 )
  {
    if ( !v3 )
      goto LABEL_12;
  }
  else
  {
    v5 = *a1;
    v3 = _byteswap_ulong(v3);
    a1[1] = v3;
    *a1 = _byteswap_ulong(v5);
    if ( !v3 )
      return;
  }
  v6 = (unsigned __int8 *)(a1 + 2);
  v7 = 0;
  do
  {
    v8 = *v6++;
    v7 += v8;
  }
  while ( v6 != (unsigned __int8 *)((char *)a1 + v3 + 8) );
  v9 = (v3 + 15) & 0xFFFFFFF8;
  if ( v7 )
  {
    v10 = (unsigned __int64 *)((char *)a1 + v9);
    v11 = (__int64)&a1[4 * (v7 - 1) + 4] + v9;
    do
    {
      v12 = *v10;
      v10 += 2;
      *(v10 - 2) = _byteswap_uint64(v12);
      *(v10 - 1) = _byteswap_uint64(*(v10 - 1));
    }
    while ( v10 != (unsigned __int64 *)v11 );
  }
  if ( a2 == 1 )
  {
    v4 = a1[1];
LABEL_12:
    a1[1] = _byteswap_ulong(v4);
    *a1 = _byteswap_ulong(*a1);
  }
}

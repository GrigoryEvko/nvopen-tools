// Function: sub_622470
// Address: 0x622470
//
__int64 __fastcall sub_622470(unsigned __int64 a1, _BYTE *a2)
{
  _BYTE *v2; // r9
  int v3; // r8d
  __int64 v4; // rcx
  unsigned __int64 v5; // rax
  char *v6; // rax
  __int64 v7; // rdi
  char v8; // dl
  char v9; // cl

  v2 = a2;
  v3 = 0;
  do
  {
    ++v2;
    v4 = v3++;
    *(v2 - 1) = a1 % 0xA + 48;
    v5 = a1;
    a1 /= 0xAu;
  }
  while ( v5 > 9 );
  a2[v3] = 0;
  if ( v3 >> 1 )
  {
    v6 = &a2[v4];
    v7 = (__int64)&a2[v4 - 1 - (unsigned int)((v3 >> 1) - 1)];
    do
    {
      v8 = *a2;
      v9 = *v6--;
      *a2++ = v9;
      v6[1] = v8;
    }
    while ( v6 != (char *)v7 );
  }
  return (unsigned int)v3;
}

// Function: sub_1249540
// Address: 0x1249540
//
unsigned __int64 __fastcall sub_1249540(_BYTE *a1, int a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rcx
  int v4; // esi
  __int64 v5; // rdx
  unsigned __int64 result; // rax
  char v7; // r11
  __int64 v8; // r10

  v3 = a3;
  v4 = a2 - 1;
  if ( a3 > 0x63 )
  {
    do
    {
      v5 = v3
         - 20 * (v3 / 0x64 + (((0x28F5C28F5C28F5C3LL * (unsigned __int128)(v3 >> 2)) >> 64) & 0xFFFFFFFFFFFFFFFCLL));
      result = v3;
      v3 /= 0x64u;
      v7 = a00010203040506_0[2 * v5 + 1];
      LOBYTE(v5) = a00010203040506_0[2 * v5];
      a1[v4] = v7;
      v8 = (unsigned int)(v4 - 1);
      v4 -= 2;
      a1[v8] = v5;
    }
    while ( result > 0x270F );
  }
  if ( v3 <= 9 )
  {
    *a1 = v3 + 48;
  }
  else
  {
    result = (unsigned __int8)a00010203040506_0[2 * v3];
    *(_WORD *)a1 = *(_WORD *)&a00010203040506_0[2 * v3];
  }
  return result;
}

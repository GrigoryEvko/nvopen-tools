// Function: sub_2554A60
// Address: 0x2554a60
//
__int64 __fastcall sub_2554A60(_BYTE *a1, int a2, unsigned int a3)
{
  unsigned int v3; // ecx
  int v4; // esi
  unsigned int v5; // eax
  unsigned int v6; // r9d
  __int64 v7; // rax
  __int64 v8; // r10
  __int64 result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx

  v3 = a3;
  v4 = a2 - 1;
  if ( a3 > 0x63 )
  {
    do
    {
      v5 = v3 % 0x64;
      v6 = v3;
      v3 /= 0x64u;
      v7 = 2 * v5;
      v8 = (unsigned int)(v7 + 1);
      result = (unsigned __int8)a00010203040506[v7];
      a1[v4] = a00010203040506[v8];
      v10 = (unsigned int)(v4 - 1);
      v4 -= 2;
      a1[v10] = result;
    }
    while ( v6 > 0x270F );
  }
  if ( v3 <= 9 )
  {
    *a1 = v3 + 48;
  }
  else
  {
    v11 = 2 * v3;
    result = (unsigned __int8)a00010203040506[v11];
    a1[1] = a00010203040506[(unsigned int)(v11 + 1)];
    *a1 = result;
  }
  return result;
}

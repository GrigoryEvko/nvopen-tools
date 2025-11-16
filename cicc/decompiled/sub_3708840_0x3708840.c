// Function: sub_3708840
// Address: 0x3708840
//
__int64 __fastcall sub_3708840(int *a1, int a2, int a3, char a4)
{
  int v4; // esi
  __int64 result; // rax

  v4 = a2 & 0xFFFFFF | ((a3 - a2) << 24) & 0x7F000000;
  result = v4 | 0x80000000;
  if ( a4 )
    v4 |= 0x80000000;
  *a1 = v4;
  return result;
}

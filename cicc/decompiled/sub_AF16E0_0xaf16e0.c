// Function: sub_AF16E0
// Address: 0xaf16e0
//
__int64 __fastcall sub_AF16E0(unsigned int a1, int *a2, int *a3, _DWORD *a4)
{
  unsigned int v6; // eax
  int v7; // edx
  unsigned int v8; // edx
  int v9; // ecx
  __int64 result; // rax

  v6 = a1 >> 1;
  if ( (a1 & 1) != 0 )
  {
    *a2 = 0;
  }
  else
  {
    v7 = (a1 >> 1) & 0x1F;
    if ( (v6 & 0x20) != 0 )
      v7 |= (a1 >> 2) & 0xFE0;
    *a2 = v7;
    v6 = a1 >> ((a1 & 0x40) == 0 ? 7 : 14);
  }
  v8 = v6 >> 1;
  if ( (v6 & 1) != 0 )
  {
    *a3 = 0;
  }
  else
  {
    v9 = (v6 >> 1) & 0x1F;
    if ( (v8 & 0x20) != 0 )
      v9 |= (v6 >> 2) & 0xFE0;
    *a3 = v9;
    v8 = v6 >> ((v6 & 0x40) == 0 ? 7 : 14);
  }
  result = 0;
  if ( (v8 & 1) == 0 )
  {
    result = (v8 >> 1) & 0x1F;
    if ( ((v8 >> 1) & 0x20) != 0 )
      result = (v8 >> 2) & 0xFE0 | (unsigned int)result;
  }
  *a4 = result;
  return result;
}

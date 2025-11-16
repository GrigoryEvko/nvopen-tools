// Function: sub_40ED43
// Address: 0x40ed43
//
__int64 __fastcall sub_40ED43(unsigned int *a1, __int64 a2, int a3)
{
  __int64 result; // rax
  int v5; // ecx
  int v6; // r8d
  int v7; // r9d

  result = *a1;
  if ( (unsigned int)result > 1 )
  {
    if ( (_DWORD)result == 2 )
    {
      sub_130F150(a1);
      result = sub_130F0B0((_DWORD)a1, (unsigned int)"%s\n", a3, v5, v6, v7);
      ++a1[6];
      *((_BYTE *)a1 + 28) = 0;
    }
  }
  else
  {
    result = sub_130F450();
    if ( *a1 <= 1 )
      return sub_130F360(a1);
  }
  return result;
}

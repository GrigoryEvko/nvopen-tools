// Function: sub_2149EA0
// Address: 0x2149ea0
//
__int64 __fastcall sub_2149EA0(_QWORD *a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v3; // r8
  unsigned __int64 v4; // rdx
  char v5; // cl
  __int64 result; // rax

  v3 = &a2[a3];
  *a1 = 0;
  a1[1] = 0;
  for ( a1[2] = 0; v3 != a2; a1[v4 >> 6] |= 1LL << v5 )
  {
    v4 = (unsigned int)*a2;
    v5 = *a2;
    if ( (unsigned int)v4 > 0xBF )
      sub_222CF80("%s: __position (which is %zu) >= _Nb (which is %zu)", (char)"bitset::set");
    ++a2;
    result = 1LL << v5;
  }
  return result;
}

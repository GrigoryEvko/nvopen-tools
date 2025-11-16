// Function: sub_2216040
// Address: 0x2216040
//
__int64 __fastcall sub_2216040(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  signed __int64 v3; // rdi
  unsigned __int64 v4; // rdx
  __int64 result; // rax

  if ( a1 > 0xFFFFFFFFFFFFFFELL )
    sub_4262D8((__int64)"basic_string::_S_create");
  v2 = a1;
  if ( a1 <= a2 )
  {
    v3 = 4 * a1 + 28;
  }
  else
  {
    if ( a1 < 2 * a2 )
      v2 = 2 * a2;
    v3 = 4 * v2 + 28;
    v4 = 4 * v2 + 60;
    if ( v4 <= 0x1000 || v2 <= a2 )
    {
      if ( v3 < 0 )
        sub_4261EA(v3, a2, v4);
    }
    else
    {
      v2 += (4096 - (v4 & 0xFFF)) >> 2;
      if ( v2 > 0xFFFFFFFFFFFFFFELL )
        v2 = 0xFFFFFFFFFFFFFFELL;
      v3 = 4 * v2 + 28;
    }
  }
  result = sub_22077B0(v3);
  *(_QWORD *)(result + 8) = v2;
  *(_DWORD *)(result + 16) = 0;
  return result;
}

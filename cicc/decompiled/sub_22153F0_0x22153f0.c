// Function: sub_22153F0
// Address: 0x22153f0
//
__int64 __fastcall sub_22153F0(unsigned __int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rdx
  unsigned __int64 v4; // rdi
  __int64 result; // rax

  if ( a1 > 0x3FFFFFFFFFFFFFF9LL )
    sub_4262D8((__int64)"basic_string::_S_create");
  v2 = a1;
  if ( a1 <= a2 )
  {
    v4 = a1 + 25;
  }
  else
  {
    if ( a1 < 2 * a2 )
      v2 = 2 * a2;
    v3 = v2 + 57;
    if ( v2 + 57 <= 0x1000 || v2 <= a2 )
    {
      v4 = v2 + 25;
      if ( (__int64)(v2 + 25) < 0 )
        sub_4261EA(v4, a2, v3);
    }
    else
    {
      v2 = v2 + 4096 - (v3 & 0xFFF);
      if ( v2 > 0x3FFFFFFFFFFFFFF9LL )
        v2 = 0x3FFFFFFFFFFFFFF9LL;
      v4 = v2 + 25;
    }
  }
  result = sub_22077B0(v4);
  *(_QWORD *)(result + 8) = v2;
  *(_DWORD *)(result + 16) = 0;
  return result;
}

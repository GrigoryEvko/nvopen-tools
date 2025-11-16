// Function: sub_721A20
// Address: 0x721a20
//
unsigned __int64 __fastcall sub_721A20(unsigned __int64 a1)
{
  int v1; // eax
  __int64 v2; // rsi
  unsigned __int64 result; // rax
  __int64 v4; // rsi

  v1 = dword_4F078D0;
  if ( !dword_4F078D0 )
  {
    v1 = j__getpagesize();
    dword_4F078D0 = v1;
  }
  v2 = v1;
  result = v1 * (a1 / (unsigned int)v1);
  v4 = result + v2;
  if ( a1 > result )
    return v4;
  return result;
}

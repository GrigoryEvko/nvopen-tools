// Function: sub_303E6A0
// Address: 0x303e6a0
//
__int64 __fastcall sub_303E6A0(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5)
{
  unsigned int v7; // eax
  unsigned int v8; // ebx
  __int64 result; // rax

  LOWORD(v7) = sub_CE9380(a2, a4);
  v8 = v7;
  result = sub_303E610(a1, a2, a3, a5);
  if ( BYTE1(v8) )
    return v8;
  return result;
}

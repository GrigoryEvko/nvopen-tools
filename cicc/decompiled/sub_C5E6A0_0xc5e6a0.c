// Function: sub_C5E6A0
// Address: 0xc5e6a0
//
unsigned __int64 __fastcall sub_C5E6A0(unsigned __int64 a1, unsigned __int8 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  unsigned __int64 v8; // rcx

  result = HIDWORD(a1);
  v8 = HIDWORD(a1);
  if ( !(_DWORD)a1 )
    sub_409306(a3, a4, a5, v8);
  if ( (_DWORD)a1 == 1 )
    sub_40930C(a3, a4, a5, v8, a2);
  return result;
}

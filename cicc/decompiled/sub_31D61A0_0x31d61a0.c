// Function: sub_31D61A0
// Address: 0x31d61a0
//
__int64 __fastcall sub_31D61A0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 208);
  if ( *(_BYTE *)(result + 291) )
    return sub_31D60D0(a1, a2);
  return result;
}

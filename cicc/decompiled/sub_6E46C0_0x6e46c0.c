// Function: sub_6E46C0
// Address: 0x6e46c0
//
__int64 __fastcall sub_6E46C0(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  sub_6E4620((_BYTE *)a1, a2);
  *(_BYTE *)(a1 + 18) = ((*(_BYTE *)(a2 + 16) & 1) << 6) | *(_BYTE *)(a1 + 18) & 0xBF;
  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 6 || (_BYTE)result == 3 )
  {
    result = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a1 + 112) = result;
  }
  return result;
}

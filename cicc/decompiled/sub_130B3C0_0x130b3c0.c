// Function: sub_130B3C0
// Address: 0x130b3c0
//
__int64 __fastcall sub_130B3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  if ( (unsigned __int8)sub_1348460(
                          (int)a2 + 62384,
                          *(_QWORD *)a2,
                          *(_QWORD *)(a2 + 68264),
                          *(_QWORD *)(a2 + 68272),
                          (int)a2 + 68096,
                          *(_DWORD *)(a2 + 68240),
                          a3) )
    return 1;
  result = sub_130EA10(a1, a2 + 62264, *(_QWORD *)(a2 + 68272), a2 + 62384, a4);
  if ( (_BYTE)result )
    return 1;
  *(_BYTE *)(a2 + 17) = 1;
  *(_BYTE *)(a2 + 16) = 1;
  return result;
}

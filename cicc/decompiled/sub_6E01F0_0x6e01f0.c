// Function: sub_6E01F0
// Address: 0x6e01f0
//
_QWORD *__fastcall sub_6E01F0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // rdx

  if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
  {
    *(_BYTE *)(a1 + 24) = 38;
    v2 = qword_4F06BB0;
    qword_4F06BB0 = a1;
    *(_QWORD *)(a1 + 80) = v2;
    return &qword_4F06BB0;
  }
  return result;
}

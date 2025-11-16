// Function: sub_878810
// Address: 0x878810
//
_QWORD *__fastcall sub_878810(__int64 a1, int a2)
{
  __int64 v2; // r12
  __int64 v3; // r13
  _QWORD *result; // rax

  v2 = a1;
  if ( a1 )
  {
    if ( (*(_BYTE *)(a1 + 124) & 1) != 0 )
      v2 = sub_735B70(a1);
    v3 = *(_QWORD *)(*(_QWORD *)v2 + 96LL);
    result = *(_QWORD **)(v3 + 152);
    if ( !result )
    {
      result = sub_8787C0();
      result[1] = v2;
      *(_QWORD *)(v3 + 152) = result;
    }
  }
  else
  {
    result = (_QWORD *)qword_4F60030;
    if ( !qword_4F60030 )
    {
      result = sub_8787C0();
      result[1] = 0;
      qword_4F60030 = (__int64)result;
    }
  }
  if ( !a2 )
  {
    result = sub_8787C0();
    result[1] = v2;
  }
  return result;
}

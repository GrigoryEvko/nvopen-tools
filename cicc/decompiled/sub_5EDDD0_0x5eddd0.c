// Function: sub_5EDDD0
// Address: 0x5eddd0
//
_QWORD *__fastcall sub_5EDDD0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  _QWORD *result; // rax

  if ( (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
  {
    v2 = *(_QWORD **)(a1 + 232);
    if ( v2 )
    {
      while ( v2[1] != a2 )
      {
        v2 = (_QWORD *)*v2;
        if ( !v2 )
          goto LABEL_3;
      }
      sub_684B00(324, dword_4F07508);
    }
  }
LABEL_3:
  v3 = (_QWORD *)sub_727D20(a1);
  v4 = 0;
  v3[1] = a2;
  if ( (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
    v4 = *(_QWORD *)(a1 + 232);
  *v3 = v4;
  *(_QWORD *)(a1 + 232) = v3;
  v5 = *(_QWORD *)(a2 + 168);
  result = (_QWORD *)sub_725220();
  result[1] = a1;
  *result = *(_QWORD *)(v5 + 136);
  *(_QWORD *)(v5 + 136) = result;
  return result;
}

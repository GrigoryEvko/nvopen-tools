// Function: sub_7E18E0
// Address: 0x7e18e0
//
_QWORD *__fastcall sub_7E18E0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r12
  _QWORD *result; // rax
  __int64 v5; // rcx
  __int64 v6; // rax

  v3 = (_QWORD *)unk_4D03F68;
  unk_4D03F68 = a1;
  *(_QWORD *)a1 = v3;
  if ( !a2 )
    a2 = v3[1];
  result = &qword_4F06BC0;
  *(_QWORD *)(a1 + 8) = a2;
  v5 = qword_4F06BC0;
  if ( !a3 )
  {
    a3 = *(_QWORD *)(a2 + 88);
    if ( a3 || !v3 )
    {
      *(_QWORD *)(a1 + 16) = a3;
      *(_QWORD *)(a1 + 56) = v5;
      *(_BYTE *)(a1 + 24) = a3 != 0;
      if ( a3 )
        goto LABEL_5;
      *(_WORD *)(a1 + 25) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      if ( !v3 )
        goto LABEL_6;
    }
    else
    {
      v6 = v3[2];
      *(_QWORD *)(a1 + 56) = qword_4F06BC0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 16) = v6;
      *(_WORD *)(a1 + 24) = 0;
      *(_BYTE *)(a1 + 26) = 0;
      *(_QWORD *)(a1 + 40) = v3[5];
    }
    result = (_QWORD *)v3[6];
    *(_QWORD *)(a1 + 48) = result;
    goto LABEL_12;
  }
  *(_QWORD *)(a1 + 16) = a3;
  *(_BYTE *)(a1 + 24) = 1;
  *(_QWORD *)(a1 + 56) = v5;
LABEL_5:
  qword_4F06BC0 = a3;
  result = 0;
  *(_WORD *)(a1 + 25) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  if ( !v3 )
  {
LABEL_6:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 32) = 0;
    *(_QWORD *)(a1 + 64) = 0;
LABEL_7:
    *(_QWORD *)(a1 + 72) = 0;
    *(_QWORD *)(a1 + 80) = 0;
    return result;
  }
  result = (_QWORD *)sub_7E18B0();
  a2 = *(_QWORD *)(a1 + 8);
LABEL_12:
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  if ( v3[1] != a2 )
    goto LABEL_7;
  *(_QWORD *)(a1 + 72) = v3[9];
  result = (_QWORD *)v3[10];
  *(_QWORD *)(a1 + 80) = result;
  return result;
}

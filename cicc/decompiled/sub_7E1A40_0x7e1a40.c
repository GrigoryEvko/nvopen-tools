// Function: sub_7E1A40
// Address: 0x7e1a40
//
__int64 __fastcall sub_7E1A40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  if ( !a2 )
  {
    a2 = *(_QWORD *)(unk_4D03F68 + 8LL);
    if ( a3 )
      goto LABEL_3;
LABEL_5:
    a3 = *(_QWORD *)(unk_4D03F68 + 16LL);
    goto LABEL_3;
  }
  if ( !a3 )
    goto LABEL_5;
LABEL_3:
  *(_QWORD *)a4 = unk_4D03F68;
  unk_4D03F68 = 0;
  sub_7E18E0(a1, a2, a3);
  result = unk_4D03F68;
  *(_QWORD *)(unk_4D03F68 + 40LL) = *(_QWORD *)(*(_QWORD *)a4 + 40LL);
  return result;
}

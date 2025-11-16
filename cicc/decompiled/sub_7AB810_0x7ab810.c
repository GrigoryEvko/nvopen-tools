// Function: sub_7AB810
// Address: 0x7ab810
//
__int64 sub_7AB810()
{
  __int64 result; // rax
  char v1; // dl

  result = qword_4F08560;
  qword_4F08560 = *(_QWORD *)qword_4F08560;
  v1 = *(_BYTE *)(result + 26);
  if ( v1 == 2 )
  {
    *(_QWORD *)(*(_QWORD *)(result + 48) + 120LL) = qword_4F08550;
    qword_4F08550 = *(_QWORD *)(result + 48);
  }
  else if ( v1 == 8 )
  {
    *(_QWORD *)(*(_QWORD *)(result + 48) + 120LL) = qword_4F08550;
    *(_QWORD *)(*(_QWORD *)(result + 56) + 120LL) = *(_QWORD *)(result + 48);
    qword_4F08550 = *(_QWORD *)(result + 56);
  }
  *(_QWORD *)result = qword_4F08558;
  qword_4F08558 = result;
  return result;
}

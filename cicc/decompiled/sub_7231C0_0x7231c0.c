// Function: sub_7231C0
// Address: 0x7231c0
//
__int64 __fastcall sub_7231C0(char *a1, int a2, int a3)
{
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 v7; // rdx

  v5 = qword_4F07928;
  if ( !qword_4F07928 )
  {
    qword_4F07928 = sub_8237A0(256);
    v5 = qword_4F07928;
  }
  sub_823800(v5);
  sub_722FC0(a1, (_QWORD *)qword_4F07928, a2, a3);
  result = qword_4F07928;
  v7 = *(_QWORD *)(qword_4F07928 + 16);
  if ( (unsigned __int64)(v7 + 1) > *(_QWORD *)(qword_4F07928 + 8) )
  {
    sub_823810(qword_4F07928);
    result = qword_4F07928;
    v7 = *(_QWORD *)(qword_4F07928 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(result + 32) + v7) = 0;
  ++*(_QWORD *)(result + 16);
  return result;
}

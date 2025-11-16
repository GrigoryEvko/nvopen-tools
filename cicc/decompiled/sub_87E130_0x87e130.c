// Function: sub_87E130
// Address: 0x87e130
//
_QWORD *__fastcall sub_87E130(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v6; // rdx
  __int64 v7; // rbx
  _QWORD *result; // rax
  _QWORD *v9; // rcx

  v6 = (_QWORD *)qword_4F5FFE0;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( qword_4F5FFE0 )
    qword_4F5FFE0 = *(_QWORD *)qword_4F5FFE0;
  else
    v6 = (_QWORD *)sub_823970(40);
  *v6 = 0;
  v6[1] = a1;
  v6[2] = a2;
  v6[3] = a3;
  v6[4] = *a4;
  result = *(_QWORD **)(v7 + 232);
  if ( result )
  {
    do
    {
      v9 = result;
      result = (_QWORD *)*result;
    }
    while ( result );
    *v9 = v6;
  }
  else
  {
    *(_QWORD *)(v7 + 232) = v6;
  }
  return result;
}

// Function: sub_80F9E0
// Address: 0x80f9e0
//
__int64 __fastcall sub_80F9E0(char *s, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rdi
  __int64 v6; // rax
  _QWORD *v7; // rdi
  __int64 result; // rax
  size_t v9; // rax

  v5 = (_QWORD *)qword_4F18BE0;
  ++*a3;
  v6 = v5[2];
  if ( (unsigned __int64)(v6 + 1) > v5[1] )
  {
    sub_823810(v5);
    v5 = (_QWORD *)qword_4F18BE0;
    v6 = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v5[4] + v6) = 76;
  ++v5[2];
  sub_80F5E0(a2, 0, a3);
  if ( *s == 45 )
    *s = 110;
  if ( !(unsigned int)sub_7E1E50(a2) )
  {
    v9 = strlen(s);
    *a3 += v9;
    sub_8238B0(qword_4F18BE0, s, v9);
  }
  v7 = (_QWORD *)qword_4F18BE0;
  ++*a3;
  result = v7[2];
  if ( (unsigned __int64)(result + 1) > v7[1] )
  {
    sub_823810(v7);
    v7 = (_QWORD *)qword_4F18BE0;
    result = *(_QWORD *)(qword_4F18BE0 + 16);
  }
  *(_BYTE *)(v7[4] + result) = 69;
  ++v7[2];
  return result;
}

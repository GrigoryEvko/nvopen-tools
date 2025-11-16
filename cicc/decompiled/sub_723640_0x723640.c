// Function: sub_723640
// Address: 0x723640
//
__int64 __fastcall sub_723640(__int64 a1, int a2, int a3)
{
  __int64 v5; // rdi
  char *v6; // r12
  _QWORD *v7; // rdi
  const char *v8; // r13
  size_t v9; // rax
  __int64 v10; // rax
  _QWORD *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 result; // rax

  v5 = qword_4F07928;
  if ( !qword_4F07928 )
  {
    qword_4F07928 = sub_8237A0(256);
    v5 = qword_4F07928;
  }
  sub_823800(v5);
  if ( a2 )
  {
    v6 = *(char **)(a1 + 16);
    v7 = (_QWORD *)qword_4F07928;
    if ( *(_QWORD *)(a1 + 64) )
    {
LABEL_5:
      sub_8238B0(v7, "module \"", 8);
      v8 = *(const char **)(*(_QWORD *)(a1 + 64) + 8LL);
      v9 = strlen(v8);
      sub_8238B0(qword_4F07928, v8, v9);
      sub_8238B0(qword_4F07928, "\" (", 3);
      v7 = (_QWORD *)qword_4F07928;
      goto LABEL_6;
    }
  }
  else
  {
    v6 = *(char **)a1;
    v7 = (_QWORD *)qword_4F07928;
    if ( *(_QWORD *)(a1 + 64) )
      goto LABEL_5;
  }
  if ( !a3 )
  {
    sub_722FC0(v6, v7, 0, 0);
    v11 = (_QWORD *)qword_4F07928;
    v13 = *(_QWORD *)(qword_4F07928 + 16);
    goto LABEL_11;
  }
LABEL_6:
  v10 = v7[2];
  if ( (unsigned __int64)(v10 + 1) > v7[1] )
  {
    sub_823810(v7);
    v7 = (_QWORD *)qword_4F07928;
    v10 = *(_QWORD *)(qword_4F07928 + 16);
  }
  *(_BYTE *)(v7[4] + v10) = 34;
  ++v7[2];
  sub_722FC0(v6, v7, 0, 0);
  v11 = (_QWORD *)qword_4F07928;
  v12 = *(_QWORD *)(qword_4F07928 + 16);
  if ( (unsigned __int64)(v12 + 1) > *(_QWORD *)(qword_4F07928 + 8) )
  {
    sub_823810(qword_4F07928);
    v11 = (_QWORD *)qword_4F07928;
    v12 = *(_QWORD *)(qword_4F07928 + 16);
  }
  *(_BYTE *)(v11[4] + v12) = 34;
  v13 = v11[2] + 1LL;
  v11[2] = v13;
LABEL_11:
  if ( *(_QWORD *)(a1 + 64) )
  {
    if ( (unsigned __int64)(v13 + 1) > v11[1] )
    {
      sub_823810(v11);
      v11 = (_QWORD *)qword_4F07928;
      v13 = *(_QWORD *)(qword_4F07928 + 16);
    }
    *(_BYTE *)(v11[4] + v13) = 41;
    v13 = v11[2] + 1LL;
    v11[2] = v13;
  }
  if ( (unsigned __int64)(v13 + 1) > v11[1] )
  {
    sub_823810(v11);
    v11 = (_QWORD *)qword_4F07928;
    v13 = *(_QWORD *)(qword_4F07928 + 16);
  }
  *(_BYTE *)(v11[4] + v13) = 0;
  result = v11[4];
  ++v11[2];
  return result;
}

// Function: sub_812220
// Address: 0x812220
//
_QWORD *__fastcall sub_812220(char a1, int a2, __int64 a3, char *a4, __int64 a5, __int64 a6, int a7, _QWORD *a8)
{
  int v8; // r10d
  __int64 v13; // r8
  _QWORD *result; // rax
  __int64 v15; // rdi
  char *v16; // r13
  size_t v17; // rax
  char *v18; // [rsp+0h] [rbp-50h]
  char *s; // [rsp+8h] [rbp-48h]
  char *sa; // [rsp+8h] [rbp-48h]
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = a2;
  if ( !a7 )
  {
    v15 = qword_4F18BE0;
    *a8 += 2LL;
    v18 = a4;
    sub_8238B0(v15, "on", 2);
    a4 = v18;
    v8 = a2;
  }
  v13 = qword_4F18BE0;
  if ( a3 )
  {
    *a8 += 2LL;
    sub_8238B0(v13, "cv", 2);
    result = (_QWORD *)sub_80F5E0(a3, 0, a8);
  }
  else
  {
    s = a4;
    if ( a4 )
    {
      *a8 += 2LL;
      sub_8238B0(v13, "li", 2);
      result = (_QWORD *)sub_80BC40(s, a8);
    }
    else
    {
      sa = (char *)qword_4F18BE0;
      v16 = sub_8094C0(a1, v8);
      v17 = strlen(v16);
      *a8 += v17;
      result = (_QWORD *)sub_8238B0(sa, v16, v17);
    }
  }
  if ( a6 )
  {
    if ( (*(_BYTE *)(a6 + 33) & 2) == 0 )
      return result;
LABEL_9:
    v21[0] = a5;
    return sub_811CB0(v21, 0, 0, a8);
  }
  if ( a5 )
    goto LABEL_9;
  return result;
}

// Function: sub_67C870
// Address: 0x67c870
//
_BYTE *__fastcall sub_67C870(unsigned int *a1, _QWORD *a2, const char *a3, const char *a4, const char *a5)
{
  _BYTE *result; // rax
  __int64 v10; // rdi
  __int64 v11; // rdi
  __int64 v12; // rax
  _QWORD *v13; // rbx
  __int64 v14; // r15
  __int64 v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rax
  int v18; // r14d
  size_t v19; // rdx
  size_t v20; // rax
  _QWORD *v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rsi
  char *v24; // rax
  char *v25; // r13
  size_t v26; // rax
  size_t v27; // rax
  char *v28; // rax
  size_t v29; // rax
  char *v30; // rax
  char *v31; // rax
  char *v32; // [rsp+8h] [rbp-78h]
  unsigned int v33; // [rsp+10h] [rbp-70h] BYREF
  int v34; // [rsp+14h] [rbp-6Ch] BYREF
  char *s1; // [rsp+18h] [rbp-68h] BYREF
  _BYTE v36[8]; // [rsp+20h] [rbp-60h] BYREF
  char *s2; // [rsp+28h] [rbp-58h] BYREF
  char s[80]; // [rsp+30h] [rbp-50h] BYREF

  result = (_BYTE *)a2[2];
  s2 = (char *)byte_3F871B3;
  if ( !result )
    result = a2;
  v10 = *((unsigned int *)result + 34);
  if ( (_DWORD)v10 && (result = (_BYTE *)sub_729E00(v10, &s2, v36, &v33, &v34), v34) )
  {
    v11 = *a1;
    s2 = (char *)byte_3F871B3;
    if ( !(_DWORD)v11 )
      return result;
  }
  else
  {
    v11 = *a1;
    if ( !(_DWORD)v11 )
      return result;
  }
  v12 = sub_729E00(v11, &s1, v36, &v33, &v34);
  v13 = (_QWORD *)qword_4D039E8;
  v14 = v12;
  if ( dword_4F073CC[0] )
  {
    v15 = *(_QWORD *)(qword_4D039E8 + 16);
    if ( (unsigned __int64)(v15 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
    {
      sub_823810(qword_4D039E8);
      v15 = v13[2];
    }
    *(_BYTE *)(v13[4] + v15) = 27;
    v16 = v13[2];
    v17 = v16 + 1;
    v13[2] = v16 + 1;
    if ( (unsigned __int64)(v16 + 2) > v13[1] )
    {
      sub_823810(v13);
      v17 = v13[2];
    }
    *(_BYTE *)(v13[4] + v17) = 7;
    ++v13[2];
    v13 = (_QWORD *)qword_4D039E8;
  }
  if ( !v34 )
  {
    v32 = s1;
    v18 = strcmp(s1, s2);
    v19 = strlen(a3);
    if ( v18 && (*v32 != 45 || v32[1]) )
    {
      sub_8238B0(v13, a3, v19);
      if ( v33 )
      {
        v28 = sub_67C860(1458);
        sub_823910(qword_4D039E8, v28);
        sprintf(s, "%lu", v33);
        v29 = strlen(s);
        sub_8238B0(qword_4D039E8, s, v29);
      }
      else
      {
        v24 = sub_67C860(2246);
        sub_823910(qword_4D039E8, v24);
      }
      if ( v33 )
      {
        v30 = sub_67C860(1459);
        sub_823910(qword_4D039E8, v30);
      }
      if ( v14 )
        v25 = (char *)sub_723640(v14, 0, 0);
      else
        v25 = (char *)sub_723260(s1);
    }
    else
    {
      sub_8238B0(v13, a3, v19);
      if ( !v33 )
      {
LABEL_17:
        v20 = strlen(a4);
        sub_8238B0(qword_4D039E8, a4, v20);
        goto LABEL_18;
      }
      v25 = s;
      v31 = sub_67C860(1458);
      sub_823910(qword_4D039E8, v31);
      sprintf(s, "%lu", v33);
    }
    v26 = strlen(v25);
    sub_8238B0(qword_4D039E8, v25, v26);
    goto LABEL_17;
  }
  v27 = strlen(a5);
  sub_8238B0(v13, a5, v27);
LABEL_18:
  result = dword_4F073CC;
  if ( dword_4F073CC[0] )
  {
    v21 = (_QWORD *)qword_4D039E8;
    v22 = *(_QWORD *)(qword_4D039E8 + 16);
    if ( (unsigned __int64)(v22 + 1) > *(_QWORD *)(qword_4D039E8 + 8) )
    {
      sub_823810(qword_4D039E8);
      v22 = v21[2];
    }
    *(_BYTE *)(v21[4] + v22) = 27;
    v23 = v21[2];
    result = (_BYTE *)(v23 + 1);
    v21[2] = v23 + 1;
    if ( (unsigned __int64)(v23 + 2) > v21[1] )
    {
      sub_823810(v21);
      result = (_BYTE *)v21[2];
    }
    result[v21[4]] = 1;
    ++v21[2];
  }
  return result;
}

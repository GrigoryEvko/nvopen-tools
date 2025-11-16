// Function: sub_67C120
// Address: 0x67c120
//
__int64 __fastcall sub_67C120(unsigned int *a1)
{
  __int64 v1; // r12
  const char *v2; // r12
  size_t v3; // rax
  unsigned __int16 v4; // r13
  unsigned int v5; // ebx
  size_t v6; // rax
  _QWORD *v7; // rdi
  __int64 v8; // rax
  __int64 result; // rax
  size_t v10; // rax
  unsigned int v11; // [rsp+8h] [rbp-58h] BYREF
  _BYTE v12[4]; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v13; // [rsp+10h] [rbp-50h] BYREF
  _BYTE v14[8]; // [rsp+18h] [rbp-48h] BYREF
  char s[64]; // [rsp+20h] [rbp-40h] BYREF

  sub_729E00(*a1, &v13, v14, &v11, v12);
  sub_8238B0(qword_4D039D8, "{\"artifactLocation\":", 20);
  v1 = v13;
  sub_8238B0(qword_4D039D8, "{\"uri\":\"file://", 15);
  v2 = (const char *)sub_722DF0(v1);
  v3 = strlen(v2);
  sub_8238B0(qword_4D039D8, v2, v3);
  sub_8238B0(qword_4D039D8, "\"}", 2);
  sub_8238B0(qword_4D039D8, ",\"region\":", 10);
  v4 = *((_WORD *)a1 + 2);
  v5 = v11;
  sub_8238B0(qword_4D039D8, "{\"startLine\":", 13);
  sprintf(s, "%lu", v5);
  v6 = strlen(s);
  sub_8238B0(qword_4D039D8, s, v6);
  if ( v4 )
  {
    sub_8238B0(qword_4D039D8, ",\"startColumn\":", 15);
    sprintf(s, "%lu", v4);
    v10 = strlen(s);
    sub_8238B0(qword_4D039D8, s, v10);
  }
  v7 = (_QWORD *)qword_4D039D8;
  v8 = *(_QWORD *)(qword_4D039D8 + 16);
  if ( (unsigned __int64)(v8 + 1) > *(_QWORD *)(qword_4D039D8 + 8) )
  {
    sub_823810();
    v7 = (_QWORD *)qword_4D039D8;
    v8 = *(_QWORD *)(qword_4D039D8 + 16);
  }
  *(_BYTE *)(v7[4] + v8) = 125;
  result = ++v7[2];
  if ( (unsigned __int64)(result + 1) > v7[1] )
  {
    sub_823810();
    v7 = (_QWORD *)qword_4D039D8;
    result = *(_QWORD *)(qword_4D039D8 + 16);
  }
  *(_BYTE *)(v7[4] + result) = 125;
  ++v7[2];
  return result;
}

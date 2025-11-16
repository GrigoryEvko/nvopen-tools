// Function: sub_822270
// Address: 0x822270
//
__int64 sub_822270()
{
  __int64 v0; // r14
  size_t v1; // rbx
  const char *v2; // r12
  __int64 v3; // rsi
  unsigned __int64 v4; // rdi
  char *v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  char *v8; // rcx
  const char *v9; // r15
  size_t v10; // rax
  char *v12; // rax
  int v13; // [rsp+0h] [rbp-90h]
  int v14; // [rsp+4h] [rbp-8Ch]
  size_t v15; // [rsp+8h] [rbp-88h]
  __int64 v16; // [rsp+18h] [rbp-78h] BYREF
  __m128i v17[7]; // [rsp+20h] [rbp-70h] BYREF

  sub_822070("__attribute__((nv_register_params))", "__nv_register_params__", 0, 0);
  sub_822070("__attribute__((noinline))", "__noinline__", 0, 0);
  if ( (_DWORD)qword_4F077B4
    && qword_4F077A0 > 0x2BF1Fu
    && unk_4D04520 | HIDWORD(qword_4F06A78) | (unsigned int)qword_4F06A78 )
  {
    sub_81B570("__attribute__((__arm_in(\"za\")))", "__arm_in", 1, 0, 1);
    sub_81B570("__attribute__((__arm_inout(\"za\")))", "__arm_inout", 1, 0, 1);
    sub_81B570("__attribute__((__arm_out(\"za\")))", "__arm_out", 1, 0, 1);
    sub_81B570("__attribute__((__arm_preserves(\"za\")))", "__arm_preserves", 1, 0, 1);
    sub_822070("__attribute__((__arm_streaming))", "__arm_streaming", 1, 0);
    sub_822070("__attribute__((__arm_streaming_compatible))", "__arm_streaming_compatible", 1, 0);
    if ( !unk_4D04784 )
      goto LABEL_4;
  }
  else if ( !unk_4D04784 )
  {
    goto LABEL_4;
  }
  qword_4F19570 = sub_822070(0, "_Pragma", 1, 0);
LABEL_4:
  unk_4D03BB8 = sub_822070(0, "defined", 1, 0);
  v0 = qword_4D04750;
  v14 = dword_4D03D1C;
  v13 = unk_4D03D20;
  dword_4F063F8 = 0;
  word_4F063FC[0] = 1;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  dword_4D03D1C = 0;
  dword_4D03D18 = 1;
  unk_4D03D20 = 1;
  if ( qword_4D04750 )
  {
    while ( 1 )
    {
      v9 = *(const char **)(v0 + 8);
      if ( !*(_DWORD *)(v0 + 16) )
        break;
      v10 = strlen(*(const char **)(v0 + 8));
      if ( !(unsigned int)sub_822080((__int64)v9, v10, &v16, v17) )
        goto LABEL_25;
      if ( v16 )
      {
        if ( (**(_BYTE **)(v16 + 88) & 2) != 0 )
LABEL_25:
          sub_684920(0x232u, (__int64)v9);
        sub_881DB0(v16);
        v0 = *(_QWORD *)v0;
        if ( !v0 )
          goto LABEL_17;
      }
      else
      {
LABEL_11:
        v0 = *(_QWORD *)v0;
        if ( !v0 )
          goto LABEL_17;
      }
    }
    if ( strchr(*(const char **)(v0 + 8), 10) )
      sub_684920(0x231u, (__int64)v9);
    qword_4D042B8 = (__int64)v9;
    v15 = strlen(v9);
    v1 = v15;
    sub_7B0E60(v15 + 6);
    v2 = (const char *)qword_4F06498;
    strcpy((char *)qword_4F06498, v9);
    v3 = 61;
    v4 = (unsigned __int64)v2;
    v5 = strchr(v2, 61);
    v8 = (char *)v15;
    if ( v5 )
    {
      v8 = (char *)&qword_4F077B4 + 4;
      if ( HIDWORD(qword_4F077B4) )
        *v5 = 32;
    }
    else
    {
      v3 = (__int64)&qword_4F077B4 + 4;
      v12 = (char *)&v2[v15];
      v6 = HIDWORD(qword_4F077B4);
      if ( HIDWORD(qword_4F077B4) )
      {
        v4 = 12576;
        strcpy(v12, " 1");
      }
      else
      {
        v3 = 12605;
        strcpy(v12, "=1");
      }
      v1 = v15 + 2;
    }
    v2[v1] = 0;
    v2[v1 + 1] = 2;
    v2[v1 + 2] = 0;
    v2[v1 + 3] = 1;
    qword_4F06460 = (_BYTE *)qword_4F06498;
    *(_DWORD *)&word_4F06480 = 0;
    sub_8200E0(v4, (unsigned int *)v3, qword_4F06498, (__int64)v8, v6, v7);
    qword_4D042B8 = 0;
    goto LABEL_11;
  }
LABEL_17:
  dword_4D03D18 = 0;
  unk_4D03D20 = v13;
  dword_4D03D1C = v14;
  dword_4F063F8 = 0;
  word_4F063FC[0] = 0;
  *(_QWORD *)dword_4F07508 = *(_QWORD *)&dword_4F063F8;
  return *(_QWORD *)&dword_4F063F8;
}

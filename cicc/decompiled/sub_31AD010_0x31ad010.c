// Function: sub_31AD010
// Address: 0x31ad010
//
__int64 __fastcall sub_31AD010(__int64 **a1)
{
  char v2; // r13
  __int64 v3; // r12
  __int64 v4; // rcx
  __int64 v5; // rdi
  char v6; // r8
  __int64 v7; // r14
  __int64 v8; // r12
  unsigned __int64 v9; // rbx
  __int64 v10; // rdx
  char v11; // al
  char v12; // r8
  bool v13; // al
  unsigned int v14; // r8d
  char v16; // al
  __int64 *v17; // rax
  __int64 v18; // rsi
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r10
  __int64 v24; // rax
  __int64 v25; // rsi
  unsigned int v26; // edx
  __int64 *v27; // rax
  __int64 v28; // r9
  __int64 v29; // rax
  int v30; // eax
  int v31; // r10d
  __int64 v32; // r12
  int v33; // eax
  int v34; // r9d
  char v35; // [rsp+Fh] [rbp-31h]
  unsigned __int8 v36; // [rsp+Fh] [rbp-31h]
  char v37; // [rsp+Fh] [rbp-31h]

  v2 = 1;
  v3 = sub_B2BE50(*a1[8]);
  if ( !sub_B6EA50(v3) )
  {
    v32 = sub_B6F970(v3);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v32 + 32LL))(
           v32,
           "loop-vectorize",
           14)
      || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v32 + 40LL))(
           v32,
           "loop-vectorize",
           14)
      || (v2 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v32 + 24LL))(
                 v32,
                 "loop-vectorize",
                 14)) != 0 )
    {
      v2 = 1;
    }
  }
  v5 = (__int64)*a1;
  v6 = 1;
  v7 = (*a1)[4];
  v8 = (*a1)[5];
  if ( v8 != v7 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v9 = *(_QWORD *)(*(_QWORD *)v7 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v9 == *(_QWORD *)v7 + 48LL )
          goto LABEL_19;
        if ( !v9 )
          BUG();
        v10 = *(unsigned __int8 *)(v9 - 24);
        if ( (unsigned int)(v10 - 30) > 0xA )
LABEL_19:
          BUG();
        if ( (_BYTE)v10 == 31 )
          break;
        sub_2AB8760(
          (__int64)"Unsupported basic block terminator",
          34,
          "loop control flow is not understood by vectorizer",
          0x31u,
          (__int64)"CFGNotUnderstood",
          16,
          a1[8],
          (__int64)*a1,
          0);
        if ( !v2 )
          return 0;
LABEL_11:
        v7 += 8;
        v6 = 0;
        if ( v8 == v7 )
        {
LABEL_12:
          v5 = (__int64)*a1;
          goto LABEL_13;
        }
      }
      if ( (*(_DWORD *)(v9 - 20) & 0x7FFFFFF) == 3 )
      {
        v37 = v6;
        v16 = sub_D48480((__int64)*a1, *(_QWORD *)(v9 - 120), v10, v4);
        v6 = v37;
        if ( !v16 )
        {
          v17 = a1[1];
          v18 = *(_QWORD *)(v9 - 56);
          v19 = *((_DWORD *)v17 + 6);
          v20 = v17[1];
          if ( !v19 )
            goto LABEL_28;
          v4 = (unsigned int)(v19 - 1);
          v21 = v4 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
          v22 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v22;
          if ( v18 == *v22 )
          {
LABEL_23:
            v24 = v22[1];
            if ( v24 && v18 == **(_QWORD **)(v24 + 32) )
              goto LABEL_5;
          }
          else
          {
            v33 = 1;
            while ( v23 != -4096 )
            {
              v34 = v33 + 1;
              v21 = v4 & (v33 + v21);
              v22 = (__int64 *)(v20 + 16LL * v21);
              v23 = *v22;
              if ( v18 == *v22 )
                goto LABEL_23;
              v33 = v34;
            }
          }
          v25 = *(_QWORD *)(v9 - 88);
          v26 = v4 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
          v27 = (__int64 *)(v20 + 16LL * v26);
          v28 = *v27;
          if ( v25 != *v27 )
          {
            v30 = 1;
            while ( v28 != -4096 )
            {
              v31 = v30 + 1;
              v26 = v4 & (v30 + v26);
              v27 = (__int64 *)(v20 + 16LL * v26);
              v28 = *v27;
              if ( *v27 == v25 )
                goto LABEL_26;
              v30 = v31;
            }
LABEL_28:
            sub_2AB8760(
              (__int64)"Unsupported conditional branch",
              30,
              "loop control flow is not understood by vectorizer",
              0x31u,
              (__int64)"CFGNotUnderstood",
              16,
              a1[8],
              (__int64)*a1,
              0);
            if ( !v2 )
              return 0;
            goto LABEL_11;
          }
LABEL_26:
          v29 = v27[1];
          if ( !v29 || **(_QWORD **)(v29 + 32) != v25 )
            goto LABEL_28;
        }
      }
LABEL_5:
      v7 += 8;
      if ( v8 == v7 )
        goto LABEL_12;
    }
  }
LABEL_13:
  v35 = v6;
  v11 = sub_31A43D0(v5, v5);
  v12 = v35;
  if ( !v11 )
  {
    sub_2AB8760(
      (__int64)"Outer loop contains divergent loops",
      35,
      "loop control flow is not understood by vectorizer",
      0x31u,
      (__int64)"CFGNotUnderstood",
      16,
      a1[8],
      (__int64)*a1,
      0);
    if ( !v2 )
      return 0;
    v12 = 0;
  }
  v36 = v12;
  v13 = sub_31ACEB0((__int64 *)a1);
  v14 = v36;
  if ( !v13 )
  {
    sub_2AB8760(
      (__int64)"Unsupported outer loop Phi(s)",
      29,
      "Unsupported outer loop Phi(s)",
      0x1Du,
      (__int64)"UnsupportedPhi",
      14,
      a1[8],
      (__int64)*a1,
      0);
    return 0;
  }
  return v14;
}

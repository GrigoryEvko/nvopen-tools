// Function: sub_398B5B0
// Address: 0x398b5b0
//
char *__fastcall sub_398B5B0(_QWORD *a1)
{
  char *v1; // rsi
  char *result; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  char *v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // r13
  void (__fastcall *v9)(__int64, _QWORD, _QWORD); // rbx
  __int64 v10; // rax
  _QWORD *v11; // r15
  _QWORD *i; // rbx
  __int64 v13; // rax
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // rdi
  __int64 *v17; // r8
  __int64 v18; // rax
  void (*v19)(); // rax
  const char *v20; // [rsp-58h] [rbp-58h] BYREF
  char v21; // [rsp-48h] [rbp-48h]
  char v22; // [rsp-47h] [rbp-47h]

  v1 = (char *)a1[70];
  result = (char *)a1[69];
  if ( v1 == result )
    return result;
  v3 = (v1 - result) >> 6;
  v4 = (v1 - result) >> 4;
  if ( v3 > 0 )
  {
    v6 = &result[64 * v3];
    while ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 1) + 80LL) + 36LL) == 3 )
    {
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 3) + 80LL) + 36LL) != 3 )
      {
        result += 16;
        if ( v1 != result )
          goto LABEL_10;
        return result;
      }
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 5) + 80LL) + 36LL) != 3 )
      {
        result += 32;
        if ( v1 != result )
          goto LABEL_10;
        return result;
      }
      if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 7) + 80LL) + 36LL) != 3 )
      {
        result += 48;
        if ( v1 != result )
          goto LABEL_10;
        return result;
      }
      result += 64;
      if ( v6 == result )
      {
        v4 = (v1 - result) >> 4;
        goto LABEL_22;
      }
    }
    goto LABEL_9;
  }
LABEL_22:
  if ( v4 == 2 )
  {
LABEL_29:
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 1) + 80LL) + 36LL) == 3 )
    {
      result += 16;
      goto LABEL_31;
    }
    goto LABEL_9;
  }
  if ( v4 != 3 )
  {
    if ( v4 != 1 )
      return result;
LABEL_31:
    if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 1) + 80LL) + 36LL) == 3 )
      return result;
    goto LABEL_9;
  }
  if ( *(_DWORD *)(*(_QWORD *)(*((_QWORD *)result + 1) + 80LL) + 36LL) == 3 )
  {
    result += 16;
    goto LABEL_29;
  }
LABEL_9:
  if ( v1 != result )
  {
LABEL_10:
    v7 = a1[1];
    v8 = *(_QWORD *)(v7 + 256);
    v9 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v8 + 160LL);
    v10 = sub_396DD80(v7);
    v9(v8, *(_QWORD *)(v10 + 168), 0);
    v11 = (_QWORD *)a1[70];
    for ( i = (_QWORD *)a1[69]; v11 != i; i += 2 )
    {
      v13 = i[1];
      if ( *(_DWORD *)(*(_QWORD *)(v13 + 80) + 36LL) != 3 )
      {
        v14 = *(_QWORD *)(v13 + 616);
        if ( !v14 )
          v14 = i[1];
        v15 = *(_QWORD *)(*i + 8 * (8LL - *(unsigned int *)(*i + 8LL)));
        if ( v15 && *(_DWORD *)(v15 + 8) )
        {
          (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1[1] + 256LL) + 176LL))(
            *(_QWORD *)(a1[1] + 256LL),
            *(_QWORD *)(v14 + 632),
            0);
          sub_398B530((__int64)a1, v15, v14);
        }
      }
    }
    v16 = a1[1];
    v17 = *(__int64 **)(v16 + 256);
    v18 = *v17;
    v22 = 1;
    v20 = "End Of Macro List Mark";
    v19 = *(void (**)())(v18 + 104);
    v21 = 3;
    if ( v19 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64 *, const char **, __int64))v19)(v17, &v20, 1);
      v16 = a1[1];
    }
    return (char *)sub_396F300(v16, 0);
  }
  return result;
}

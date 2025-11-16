// Function: sub_31A6C80
// Address: 0x31a6c80
//
__int64 __fastcall sub_31A6C80(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r15
  __int64 v8; // rsi
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 v11; // rdx
  __int64 v13; // r13

  v3 = 1;
  v4 = sub_B2BE50(**(_QWORD **)(a1 + 64));
  if ( !sub_B6EA50(v4) )
  {
    v13 = sub_B6F970(v4);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v13 + 32LL))(
           v13,
           "loop-vectorize",
           14)
      || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v13 + 40LL))(
           v13,
           "loop-vectorize",
           14)
      || (v3 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v13 + 24LL))(
                 v13,
                 "loop-vectorize",
                 14),
          (_BYTE)v3) )
    {
      v3 = 1;
    }
  }
  if ( sub_D4B130(a2) )
  {
    v3 = 1;
  }
  else
  {
    sub_2AB8760(
      (__int64)"Loop doesn't have a legal pre-header",
      36,
      "loop control flow is not understood by vectorizer",
      0x31u,
      (__int64)"CFGNotUnderstood",
      16,
      *(__int64 **)(a1 + 64),
      *(_QWORD *)a1,
      0);
    if ( !(_BYTE)v3 )
      return v3;
    v3 = 0;
  }
  v5 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 16LL);
  if ( !v5 )
    goto LABEL_18;
  while ( 1 )
  {
    v6 = *(_QWORD *)(v5 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v6 - 30) <= 0xAu )
      break;
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      goto LABEL_18;
  }
  v7 = 0;
  v8 = *(_QWORD *)(v6 + 40);
  if ( !*(_BYTE *)(a2 + 84) )
    goto LABEL_15;
LABEL_7:
  v9 = *(_QWORD **)(a2 + 64);
  v10 = &v9[*(unsigned int *)(a2 + 76)];
  if ( v9 != v10 )
  {
    while ( v8 != *v9 )
    {
      if ( v10 == ++v9 )
        goto LABEL_12;
    }
    ++v7;
  }
LABEL_12:
  while ( 1 )
  {
    v5 = *(_QWORD *)(v5 + 8);
    if ( !v5 )
      break;
    v11 = *(_QWORD *)(v5 + 24);
    if ( (unsigned __int8)(*(_BYTE *)v11 - 30) <= 0xAu )
    {
      v8 = *(_QWORD *)(v11 + 40);
      if ( *(_BYTE *)(a2 + 84) )
        goto LABEL_7;
LABEL_15:
      if ( sub_C8CA60(a2 + 56, v8) )
        ++v7;
    }
  }
  if ( (_DWORD)v7 != 1 )
  {
LABEL_18:
    v3 = 0;
    sub_2AB8760(
      (__int64)"The loop must have a single backedge",
      36,
      "loop control flow is not understood by vectorizer",
      0x31u,
      (__int64)"CFGNotUnderstood",
      16,
      *(__int64 **)(a1 + 64),
      *(_QWORD *)a1,
      0);
  }
  return v3;
}

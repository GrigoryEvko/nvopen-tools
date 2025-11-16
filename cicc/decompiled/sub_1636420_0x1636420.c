// Function: sub_1636420
// Address: 0x1636420
//
__int64 __fastcall sub_1636420(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rdi
  __int64 v7; // r12
  const char *(__fastcall *v8)(__int64, __int64); // rax
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  const char *v12; // rsi
  size_t v13; // r13
  __int64 result; // rax
  _BYTE *v15; // rdi
  size_t v16; // rdx

  v4 = sub_16BA580(a1, a2, a3);
  v5 = (unsigned int)(2 * a2);
  v6 = v4;
  v7 = sub_16E8750(v4, v5);
  v8 = *(const char *(__fastcall **)(__int64, __int64))(*a1 + 16LL);
  if ( v8 == sub_1635FB0 )
  {
    v9 = a1[2];
    v10 = sub_163A1D0(v6, v5);
    v11 = sub_163A340(v10, v9);
    if ( !v11 )
    {
      v15 = *(_BYTE **)(v7 + 24);
      v13 = 43;
      v12 = "Unnamed pass: implement Pass::getPassName()";
      if ( *(_QWORD *)(v7 + 16) - (_QWORD)v15 > 0x2Au )
        goto LABEL_9;
      goto LABEL_11;
    }
    v12 = *(const char **)v11;
    v13 = *(_QWORD *)(v11 + 8);
  }
  else
  {
    v12 = (const char *)((__int64 (__fastcall *)(_QWORD *))v8)(a1);
    v13 = v16;
  }
  result = *(_QWORD *)(v7 + 16);
  v15 = *(_BYTE **)(v7 + 24);
  if ( result - (__int64)v15 >= v13 )
  {
    if ( !v13 )
      goto LABEL_6;
LABEL_9:
    memcpy(v15, v12, v13);
    result = *(_QWORD *)(v7 + 16);
    v15 = (_BYTE *)(v13 + *(_QWORD *)(v7 + 24));
    *(_QWORD *)(v7 + 24) = v15;
    if ( (_BYTE *)result != v15 )
      goto LABEL_7;
    return sub_16E7EE0(v7, "\n", 1);
  }
LABEL_11:
  v7 = sub_16E7EE0(v7, v12, v13);
  result = *(_QWORD *)(v7 + 16);
  v15 = *(_BYTE **)(v7 + 24);
LABEL_6:
  if ( (_BYTE *)result != v15 )
  {
LABEL_7:
    *v15 = 10;
    ++*(_QWORD *)(v7 + 24);
    return result;
  }
  return sub_16E7EE0(v7, "\n", 1);
}

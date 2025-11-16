// Function: sub_BB8E60
// Address: 0xbb8e60
//
__int64 __fastcall sub_BB8E60(__int64 a1, int a2)
{
  __int64 v3; // rsi
  __int64 v4; // rdi
  __int64 v5; // r12
  const char *(__fastcall *v6)(__int64, __int64); // rax
  pthread_rwlock_t *v7; // rax
  __int64 v8; // rax
  const char *v9; // rsi
  size_t v10; // r13
  __int64 result; // rax
  _BYTE *v12; // rdi
  size_t v13; // rdx

  v3 = (unsigned int)(2 * a2);
  v4 = sub_C5F790(a1);
  v5 = sub_CB69B0(v4, v3);
  v6 = *(const char *(__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 16LL);
  if ( v6 == sub_BB8680 )
  {
    v7 = (pthread_rwlock_t *)sub_BC2B00(v4, v3);
    v8 = sub_BC2C30(v7);
    if ( !v8 )
    {
      v12 = *(_BYTE **)(v5 + 32);
      v10 = 43;
      v9 = "Unnamed pass: implement Pass::getPassName()";
      if ( *(_QWORD *)(v5 + 24) - (_QWORD)v12 > 0x2Au )
        goto LABEL_9;
      goto LABEL_11;
    }
    v9 = *(const char **)v8;
    v10 = *(_QWORD *)(v8 + 8);
  }
  else
  {
    v9 = (const char *)((__int64 (__fastcall *)(__int64))v6)(a1);
    v10 = v13;
  }
  result = *(_QWORD *)(v5 + 24);
  v12 = *(_BYTE **)(v5 + 32);
  if ( result - (__int64)v12 >= v10 )
  {
    if ( !v10 )
      goto LABEL_6;
LABEL_9:
    memcpy(v12, v9, v10);
    result = *(_QWORD *)(v5 + 24);
    v12 = (_BYTE *)(v10 + *(_QWORD *)(v5 + 32));
    *(_QWORD *)(v5 + 32) = v12;
    if ( (_BYTE *)result != v12 )
      goto LABEL_7;
    return sub_CB6200(v5, "\n", 1);
  }
LABEL_11:
  v5 = sub_CB6200(v5, v9, v10);
  result = *(_QWORD *)(v5 + 24);
  v12 = *(_BYTE **)(v5 + 32);
LABEL_6:
  if ( (_BYTE *)result != v12 )
  {
LABEL_7:
    *v12 = 10;
    ++*(_QWORD *)(v5 + 32);
    return result;
  }
  return sub_CB6200(v5, "\n", 1);
}

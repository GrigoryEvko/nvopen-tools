// Function: sub_C521F0
// Address: 0xc521f0
//
_QWORD *__fastcall sub_C521F0(__int64 **a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 *v3; // rax
  __int64 *v4; // r13
  const void *v5; // r14
  size_t v6; // r12
  unsigned int v7; // eax
  unsigned int v8; // r8d
  _QWORD *v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // r8d
  _QWORD *v16; // rcx
  _QWORD *v17; // r13
  __int64 v18; // r12
  __int64 v19; // r13
  unsigned int v20; // eax
  _QWORD *result; // rax
  __int64 v22; // rdx
  _QWORD *v23; // r12
  _QWORD *v24; // [rsp+8h] [rbp-68h]
  unsigned int v25; // [rsp+14h] [rbp-5Ch]
  __int64 v26; // [rsp+18h] [rbp-58h]

  v2 = a2 + 128;
  v3 = a1[1];
  v4 = a1[2];
  v5 = (const void *)*v3;
  v6 = v3[1];
  v26 = **a1;
  v7 = sub_C92610(*v3, v6);
  v8 = sub_C92740(a2 + 128, v5, v6, v7);
  v9 = (_QWORD *)(*(_QWORD *)(a2 + 128) + 8LL * v8);
  if ( *v9 )
  {
    if ( *v9 != -8 )
    {
      v10 = sub_CB72A0(v2, v5);
      v11 = sub_CB6200(v10, *v4, v4[1]);
      v12 = sub_904010(v11, ": CommandLine Error: Option '");
      v13 = sub_A51340(v12, *(const void **)(v26 + 24), *(_QWORD *)(v26 + 32));
      sub_904010(v13, "' registered more than once!\n");
      sub_C64ED0("inconsistency in registered CommandLine options", 1);
    }
    --*(_DWORD *)(a2 + 144);
  }
  v24 = v9;
  v25 = v8;
  v14 = sub_C7D670(v6 + 17, 8);
  v15 = v25;
  v16 = v24;
  v17 = (_QWORD *)v14;
  if ( v6 )
  {
    memcpy((void *)(v14 + 16), v5, v6);
    v15 = v25;
    v16 = v24;
  }
  *((_BYTE *)v17 + v6 + 16) = 0;
  *v17 = v6;
  v17[1] = v26;
  *v16 = v17;
  ++*(_DWORD *)(a2 + 140);
  sub_C929D0(v2, v15);
  v18 = *(_QWORD *)(v26 + 32);
  v19 = *(_QWORD *)(v26 + 24);
  v20 = sub_C92610(v19, v18);
  result = (_QWORD *)sub_C92860(v2, v19, v18, v20);
  if ( (_DWORD)result != -1 )
  {
    v22 = *(_QWORD *)(a2 + 128);
    result = (_QWORD *)(v22 + 8LL * (int)result);
    if ( result != (_QWORD *)(v22 + 8LL * *(unsigned int *)(a2 + 136)) )
    {
      v23 = (_QWORD *)*result;
      sub_C929B0(v2, *result);
      return (_QWORD *)sub_C7D6A0(v23, *v23 + 17LL, 8);
    }
  }
  return result;
}

// Function: sub_2E5F5F0
// Address: 0x2e5f5f0
//
_QWORD *__fastcall sub_2E5F5F0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  void *v7; // rdx
  __int64 v8; // r15
  __int64 v9; // rdx
  _BYTE *v10; // rax
  _QWORD *result; // rax
  __int64 *v12; // rbx
  __int64 v13; // r9
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r15
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  _BYTE *v19; // rax
  __int64 v20; // [rsp+0h] [rbp-60h]
  __int64 *v21; // [rsp+8h] [rbp-58h]
  __int64 v22[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 (__fastcall *v23)(const __m128i **, const __m128i *, int); // [rsp+20h] [rbp-40h]
  __int64 (__fastcall *v24)(__int64 *, __int64); // [rsp+28h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v4) <= 5 )
  {
    v5 = sub_CB6200(a2, "depth=", 6u);
  }
  else
  {
    *(_DWORD *)v4 = 1953523044;
    v5 = a2;
    *(_WORD *)(v4 + 4) = 15720;
    *(_QWORD *)(a2 + 32) += 6LL;
  }
  v6 = sub_CB59D0(v5, *(unsigned int *)(*a1 + 168));
  v7 = *(void **)(v6 + 32);
  v8 = v6;
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 9u )
  {
    v8 = sub_CB6200(v6, ": entries(", 0xAu);
  }
  else
  {
    qmemcpy(v7, ": entries(", 10);
    *(_QWORD *)(v6 + 32) += 10LL;
  }
  v9 = *a1;
  v22[1] = a1[1];
  v22[0] = v9;
  v23 = sub_2E5D790;
  v24 = sub_2E5DC10;
  sub_2E5DC10(v22, v8);
  v10 = *(_BYTE **)(v8 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v8 + 24) )
  {
    sub_CB5D20(v8, 41);
  }
  else
  {
    *(_QWORD *)(v8 + 32) = v10 + 1;
    *v10 = 41;
  }
  if ( v23 )
    v23((const __m128i **)v22, (const __m128i *)v22, 3);
  result = (_QWORD *)*a1;
  v12 = *(__int64 **)(*a1 + 88);
  v21 = &v12[*(unsigned int *)(*a1 + 96)];
  if ( v21 != v12 )
  {
    while ( 1 )
    {
      v16 = *v12;
      v22[0] = *v12;
      v17 = (_QWORD *)result[1];
      v18 = (__int64)&v17[*((unsigned int *)result + 4)];
      result = sub_2E5D7F0(v17, v18, v22);
      if ( (_QWORD *)v18 == result )
      {
        v19 = *(_BYTE **)(a2 + 32);
        if ( (unsigned __int64)v19 < *(_QWORD *)(a2 + 24) )
        {
          v13 = a2;
          *(_QWORD *)(a2 + 32) = v19 + 1;
          *v19 = 32;
        }
        else
        {
          v13 = sub_CB5D20(a2, 32);
        }
        v14 = a1[1];
        v20 = v13;
        sub_2EE72D0(v22, v14, v16);
        if ( !v23 )
          sub_4263D6(v22, v14, v15);
        v24(v22, v20);
        result = v23;
        if ( v23 )
          result = (_QWORD *)v23((const __m128i **)v22, (const __m128i *)v22, 3);
      }
      if ( v21 == ++v12 )
        break;
      result = (_QWORD *)*a1;
    }
  }
  return result;
}

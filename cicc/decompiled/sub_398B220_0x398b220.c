// Function: sub_398B220
// Address: 0x398b220
//
__int64 __fastcall sub_398B220(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r12
  void (__fastcall *v4)(__int64, _QWORD, _QWORD); // rbx
  __int64 v5; // rax
  _QWORD *v6; // r14
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // r12
  __int64 v10; // r12
  _QWORD *v11; // rbx
  _QWORD *v12; // r12
  unsigned int v13; // eax
  __int64 v14; // rsi
  _QWORD *i; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(v2 + 256);
  v4 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v3 + 160LL);
  v5 = sub_396DD80(v2);
  v4(v3, *(_QWORD *)(v5 + 264), 0);
  v6 = *(_QWORD **)(a1 + 1192);
  result = (__int64)&v6[4 * *(unsigned int *)(a1 + 1200)];
  for ( i = (_QWORD *)result; i != v6; result = sub_396F300(*(_QWORD *)(a1 + 8), 0) )
  {
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 176LL))(
      *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
      v6[1],
      0);
    v8 = v6[2];
    if ( (((__int64)v6 - *(_QWORD *)(a1 + 1192)) >> 5) + 1 == *(_DWORD *)(a1 + 1200) )
      v9 = *(unsigned int *)(a1 + 1344);
    else
      v9 = v6[6];
    v10 = v9 - v8;
    v11 = (_QWORD *)(*(_QWORD *)(a1 + 1336) + 32 * v8);
    v12 = &v11[4 * v10];
    while ( v12 != v11 )
    {
      sub_396F300(*(_QWORD *)(a1 + 8), 3);
      v13 = sub_39BFF80(a1 + 5512, *v11, 0);
      sub_397C0C0(*(_QWORD *)(a1 + 8), v13, 0);
      sub_396F380(*(_QWORD *)(a1 + 8));
      v14 = (__int64)v11;
      v11 += 4;
      sub_398AF30(a1, v14);
    }
    v6 += 4;
  }
  return result;
}

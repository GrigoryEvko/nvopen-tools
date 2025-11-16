// Function: sub_398C990
// Address: 0x398c990
//
__int64 __fastcall sub_398C990(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v2; // r12
  __int64 v4; // rax
  char v5; // r14
  __int64 v6; // rdi
  void (__fastcall *v7)(__int64, __int64, _QWORD); // r15
  __int64 v8; // rax
  __int64 v9; // r8
  __int64 v10; // rsi
  __int64 v11; // rbx
  __int64 v12; // rdi
  void (__fastcall *v13)(__int64, _QWORD, _QWORD); // r15
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-40h]
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+0h] [rbp-40h]
  __int64 i; // [rsp+8h] [rbp-38h]

  result = a1[70];
  v2 = a1[69];
  for ( i = result; i != v2; v2 += 16 )
  {
    v11 = *(_QWORD *)(v2 + 8);
    result = sub_39C8520(v11);
    if ( (_BYTE)result )
    {
      v12 = a1[1];
      v18 = *(_QWORD *)(v12 + 256);
      v13 = *(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v18 + 160LL);
      if ( *(_BYTE *)(*(_QWORD *)(v11 + 80) + 50LL) )
      {
        v4 = sub_396DD80(v12);
        v13(v18, *(_QWORD *)(v4 + 328), 0);
        v5 = 1;
        sub_398C140((__int64)a1, 1, (__int64)"Names", 5, v11, (__int64 **)(v11 + 672));
        v6 = a1[1];
        v17 = *(_QWORD *)(v6 + 256);
        v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 160LL);
        v8 = sub_396DD80(v6);
        v9 = v17;
        v10 = *(_QWORD *)(v8 + 336);
      }
      else
      {
        v14 = sub_396DD80(v12);
        v13(v18, *(_QWORD *)(v14 + 176), 0);
        v5 = 0;
        sub_398C140((__int64)a1, 0, (__int64)"Names", 5, v11, (__int64 **)(v11 + 672));
        v15 = a1[1];
        v19 = *(_QWORD *)(v15 + 256);
        v7 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v19 + 160LL);
        v16 = sub_396DD80(v15);
        v9 = v19;
        v10 = *(_QWORD *)(v16 + 120);
      }
      v7(v9, v10, 0);
      result = sub_398C140((__int64)a1, v5, (__int64)"Types", 5, v11, (__int64 **)(v11 + 704));
    }
  }
  return result;
}

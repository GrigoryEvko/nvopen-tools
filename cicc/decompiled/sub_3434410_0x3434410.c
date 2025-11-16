// Function: sub_3434410
// Address: 0x3434410
//
void __fastcall sub_3434410(__int64 *a1, __m128i a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, int a7)
{
  __int64 v7; // r12
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // edx
  unsigned __int8 *v12; // rbx
  int v13; // r13d
  __int64 v14; // [rsp+20h] [rbp-30h] BYREF
  int v15; // [rsp+28h] [rbp-28h]

  v7 = a1[108];
  if ( (*(_BYTE *)(*(_QWORD *)v7 + 877LL) & 2) != 0 )
  {
    v8 = *((_DWORD *)a1 + 212);
    v9 = *a1;
    v14 = 0;
    v15 = v8;
    if ( v9 )
    {
      if ( &v14 != (__int64 *)(v9 + 48) )
      {
        v10 = *(_QWORD *)(v9 + 48);
        v14 = v10;
        if ( v10 )
          sub_B96E90((__int64)&v14, v10, 1);
      }
    }
    v12 = sub_33FAF80(v7, 331, (__int64)&v14, 1, 0, a7, a2);
    v13 = v11;
    if ( v12 )
    {
      nullsub_1875();
      *(_QWORD *)(v7 + 384) = v12;
      *(_DWORD *)(v7 + 392) = v13;
      sub_33E2B60();
    }
    else
    {
      *(_QWORD *)(v7 + 384) = 0;
      *(_DWORD *)(v7 + 392) = v11;
    }
    if ( v14 )
      sub_B91220((__int64)&v14, v14);
  }
}

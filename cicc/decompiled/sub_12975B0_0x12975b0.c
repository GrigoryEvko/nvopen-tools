// Function: sub_12975B0
// Address: 0x12975b0
//
__int64 __fastcall sub_12975B0(_QWORD **a1, __int64 a2, __int64 a3, __int64 *a4, __int64 *a5, __int64 *a6)
{
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 *v11; // r14
  __int64 v12; // rsi
  _QWORD *v13; // rsi
  __int64 v14; // r13
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v23; // [rsp+40h] [rbp-F0h]
  __int64 v24; // [rsp+48h] [rbp-E8h]
  __int64 v25; // [rsp+50h] [rbp-E0h]
  __int64 v26; // [rsp+58h] [rbp-D8h]
  __int64 v27; // [rsp+68h] [rbp-C8h] BYREF
  _QWORD v28[2]; // [rsp+70h] [rbp-C0h] BYREF
  _BYTE v29[176]; // [rsp+80h] [rbp-B0h] BYREF

  v6 = *a6;
  v7 = *a5;
  v8 = *((unsigned int *)a4 + 2);
  v28[0] = v29;
  v28[1] = 0x2000000000LL;
  v9 = *a4;
  v23 = v6 + *((unsigned int *)a6 + 2);
  v24 = v7 + *((unsigned int *)a5 + 2);
  v10 = *a4 + v8;
  v11 = *(__int64 **)a3;
  v25 = v10;
  v26 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
  sub_16BD4C0(v28, a2);
  if ( (__int64 *)v26 != v11 && v9 != v25 && v6 != v23 && v7 != v24 )
  {
    do
    {
      v12 = *v11++;
      ++v9;
      ++v7;
      ++v6;
      sub_16BD4C0(v28, v12);
      sub_16BD430(v28, *(unsigned __int8 *)(v9 - 1));
      sub_16BD430(v28, *(unsigned __int8 *)(v7 - 1));
      sub_16BD430(v28, *(unsigned __int8 *)(v6 - 1));
    }
    while ( (__int64 *)v26 != v11 && v25 != v9 && v24 != v7 && v23 != v6 );
  }
  v13 = v28;
  v27 = 0;
  v14 = sub_16BDDE0(a1 + 17, v28, &v27);
  if ( !v14 )
  {
    v16 = sub_22077B0(24);
    v14 = v16;
    if ( v16 )
      sub_1297470(v16, a2, a3, a4, a5, a6);
    sub_16BDA20(a1 + 17, v14, v27);
    v17 = sub_12A6160(a1);
    v13 = (_QWORD *)v14;
    (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD **))(*(_QWORD *)v17 + 16LL))(v17, v14, **a1, a1);
  }
  if ( (_BYTE *)v28[0] != v29 )
    _libc_free(v28[0], v13);
  return v14;
}

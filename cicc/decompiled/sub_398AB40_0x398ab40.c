// Function: sub_398AB40
// Address: 0x398ab40
//
__int64 __fastcall sub_398AB40(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v5; // r12
  __int64 v6; // r9
  __int64 v7; // r10
  unsigned int v8; // ebx
  unsigned int v9; // r15d
  unsigned int v10; // edx
  _QWORD *v11; // rax
  char v12; // al
  unsigned __int16 v13; // ax
  __int64 *v14; // rbx
  const __m128i *v15; // rbx
  const __m128i *v16; // r13
  __m128i v17; // xmm0
  __int64 v19; // [rsp+8h] [rbp-F8h]
  __int64 v20; // [rsp+10h] [rbp-F0h]
  _QWORD v22[3]; // [rsp+20h] [rbp-E0h] BYREF
  char v23; // [rsp+38h] [rbp-C8h]
  __m128i v24; // [rsp+40h] [rbp-C0h] BYREF
  __m128i v25; // [rsp+50h] [rbp-B0h]
  _BYTE v26[8]; // [rsp+60h] [rbp-A0h] BYREF
  char *v27; // [rsp+68h] [rbp-98h]
  char v28; // [rsp+78h] [rbp-88h] BYREF

  v5 = *a3;
  v6 = a1[1];
  v7 = *a1;
  v8 = *(_DWORD *)(*a3 + 1464);
  v9 = *(_DWORD *)(*a3 + 1192);
  v10 = *(_DWORD *)(*a3 + 152);
  if ( v10 >= *(_DWORD *)(v5 + 156) )
  {
    v19 = *a1;
    v20 = a1[1];
    sub_16CD150(v5 + 144, (const void *)(v5 + 160), 0, 32, a5, v6);
    v10 = *(_DWORD *)(v5 + 152);
    v7 = v19;
    v6 = v20;
  }
  v11 = (_QWORD *)(*(_QWORD *)(v5 + 144) + 32LL * v10);
  if ( v11 )
  {
    *v11 = v7;
    v11[2] = v9;
    v11[1] = v6;
    v11[3] = v8;
    v10 = *(_DWORD *)(v5 + 152);
  }
  v12 = *(_BYTE *)(v5 + 2496);
  *(_DWORD *)(v5 + 152) = v10 + 1;
  v23 = v12;
  v22[0] = &unk_4A3F938;
  v22[1] = v5 + 1184;
  v22[2] = v5 + 1456;
  v13 = sub_3971A70(a2);
  sub_39886F0((__int64)v26, v13, (__int64)v22, a2);
  v14 = (__int64 *)a1[2];
  sub_15B1350((__int64)&v24, *(unsigned __int64 **)(*v14 + 24), *(unsigned __int64 **)(*v14 + 32));
  if ( v25.m128i_i8[0] )
  {
    v15 = (const __m128i *)a1[2];
    v16 = &v15[2 * *((unsigned int *)a1 + 6)];
    while ( v16 != v15 )
    {
      v17 = _mm_loadu_si128(v15);
      v15 += 2;
      v24 = v17;
      v25 = _mm_loadu_si128(v15 - 1);
      sub_3984CB0(a2, a4, v24.m128i_i64, (__int64)v26);
    }
  }
  else
  {
    sub_3984CB0(a2, a4, v14, (__int64)v26);
  }
  sub_399FD30(v26);
  if ( v27 != &v28 )
    _libc_free((unsigned __int64)v27);
  return sub_39C2CC0(v5);
}

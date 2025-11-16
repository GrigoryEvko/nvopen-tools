// Function: sub_35BCFD0
// Address: 0x35bcfd0
//
__int64 __fastcall sub_35BCFD0(_QWORD *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // r15
  unsigned int v6; // r13d
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r15
  volatile signed __int32 *v11; // rdi
  __int64 v12; // rbx
  __int64 v13; // rax
  _QWORD *v14; // rdi
  _BYTE *v15; // rsi
  __int64 v16; // r14
  __int64 v17; // rax
  _QWORD *v18; // rdi
  _BYTE *v19; // rsi
  __int64 v20; // r12
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rax
  __m128i v25; // xmm0
  _DWORD v26[13]; // [rsp+Ch] [rbp-34h] BYREF

  v4 = a1[30];
  v5 = a1[26];
  if ( a1[29] == v4 )
  {
    v22 = a1[27];
    v6 = -1431655765 * ((v22 - v5) >> 4);
    v9 = 48LL * v6;
    if ( v22 == a1[28] )
    {
      sub_35BCCE0(a1 + 26, (const __m128i *)v22, (__int64 *)a2);
      v5 = a1[26];
    }
    else
    {
      if ( v22 )
      {
        v23 = *(_QWORD *)a2;
        *(_QWORD *)(v22 + 8) = 0;
        *(_QWORD *)a2 = 0;
        *(_QWORD *)v22 = v23;
        v24 = *(_QWORD *)(a2 + 8);
        *(_QWORD *)(a2 + 8) = 0;
        *(_QWORD *)(v22 + 8) = v24;
        v25 = _mm_loadu_si128((const __m128i *)(a2 + 32));
        *(_QWORD *)(v22 + 20) = *(_QWORD *)(a2 + 20);
        *(__m128i *)(v22 + 32) = v25;
        v22 = a1[27];
        v5 = a1[26];
      }
      a1[27] = v22 + 48;
    }
  }
  else
  {
    v6 = *(_DWORD *)(v4 - 4);
    a1[30] = v4 - 4;
    v7 = *(_QWORD *)a2;
    *(_QWORD *)a2 = 0;
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(a2 + 8) = 0;
    v9 = 48LL * v6;
    v10 = v9 + v5;
    v11 = *(volatile signed __int32 **)(v10 + 8);
    *(_QWORD *)v10 = v7;
    *(_QWORD *)(v10 + 8) = v8;
    if ( v11 )
      sub_A191D0(v11);
    *(_DWORD *)(v10 + 20) = *(_DWORD *)(a2 + 20);
    *(_DWORD *)(v10 + 24) = *(_DWORD *)(a2 + 24);
    *(_QWORD *)(v10 + 32) = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(v10 + 40) = *(_QWORD *)(a2 + 40);
    v5 = a1[26];
  }
  v12 = v5 + v9;
  v13 = *(unsigned int *)(v12 + 20);
  v26[0] = v6;
  v14 = (_QWORD *)(a1[20] + 96 * v13);
  v15 = (_BYTE *)v14[10];
  v16 = (__int64)&v15[-v14[9]] >> 2;
  if ( v15 == (_BYTE *)v14[11] )
  {
    sub_B8BBF0((__int64)(v14 + 9), v15, v26);
  }
  else
  {
    if ( v15 )
    {
      *(_DWORD *)v15 = v6;
      v15 = (_BYTE *)v14[10];
    }
    v14[10] = v15 + 4;
  }
  v17 = *(unsigned int *)(v12 + 24);
  *(_QWORD *)(v12 + 32) = v16;
  v26[0] = v6;
  v18 = (_QWORD *)(a1[20] + 96 * v17);
  v19 = (_BYTE *)v18[10];
  v20 = (__int64)&v19[-v18[9]] >> 2;
  if ( v19 == (_BYTE *)v18[11] )
  {
    sub_B8BBF0((__int64)(v18 + 9), v19, v26);
  }
  else
  {
    if ( v19 )
    {
      *(_DWORD *)v19 = v6;
      v19 = (_BYTE *)v18[10];
    }
    v18[10] = v19 + 4;
  }
  *(_QWORD *)(v12 + 40) = v20;
  return v6;
}

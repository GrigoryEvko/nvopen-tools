// Function: sub_3417A70
// Address: 0x3417a70
//
__int64 __fastcall sub_3417A70(const __m128i *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned int v4; // r13d
  __int64 v5; // r9
  __int64 v6; // r8
  unsigned __int64 *v7; // rdx
  unsigned int v8; // r15d
  unsigned __int64 **v9; // rdi
  __m128i v10; // xmm0
  __int64 v11; // rax
  __m128i *v12; // rax
  int v13; // edx
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r13
  __m128i v21; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v22; // [rsp-84h] [rbp-84h]
  unsigned __int64 **v23; // [rsp-80h] [rbp-80h]
  unsigned __int64 *v24; // [rsp-78h] [rbp-78h] BYREF
  __int64 v25; // [rsp-70h] [rbp-70h]
  _BYTE v26[104]; // [rsp-68h] [rbp-68h] BYREF

  v2 = (unsigned int)(*(_DWORD *)(a2 + 24) - 101);
  if ( (unsigned int)v2 > 0x2F )
    BUG();
  v4 = (unsigned __int16)aAbcd_0[v2];
  sub_34161C0((__int64)a1, a2, 1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = *(unsigned int *)(a2 + 64);
  v24 = (unsigned __int64 *)v26;
  v25 = 0x300000000LL;
  if ( (_DWORD)v6 != 1 )
  {
    v7 = (unsigned __int64 *)v26;
    v8 = 1;
    v9 = &v24;
    v10 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
    v11 = 0;
    while ( 1 )
    {
      ++v8;
      *(__m128i *)&v7[2 * v11] = v10;
      v11 = (unsigned int)(v25 + 1);
      LODWORD(v25) = v25 + 1;
      if ( (_DWORD)v6 == v8 )
        break;
      v10 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL * v8));
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v25) )
      {
        v22 = v6;
        v23 = v9;
        v21 = v10;
        sub_C8D5F0((__int64)v9, v26, v11 + 1, 0x10u, v6, v5);
        v11 = (unsigned int)v25;
        v6 = v22;
        v10 = _mm_load_si128(&v21);
        v9 = v23;
      }
      v7 = v24;
    }
  }
  v12 = sub_33ED250((__int64)a1, **(unsigned __int16 **)(a2 + 48), *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL));
  v15 = sub_33EC480((__int64)a1, a2, v4, (unsigned __int64)v12, v13, v14, v24, (unsigned int)v25);
  v19 = v15;
  if ( a2 == v15 )
  {
    *(_DWORD *)(v15 + 36) = -1;
  }
  else
  {
    sub_34158F0((__int64)a1, a2, v15, v16, v17, v18);
    sub_33ECEA0(a1, a2);
  }
  if ( v24 != (unsigned __int64 *)v26 )
    _libc_free((unsigned __int64)v24);
  return v19;
}

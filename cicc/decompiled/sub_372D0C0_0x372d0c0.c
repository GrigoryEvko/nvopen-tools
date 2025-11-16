// Function: sub_372D0C0
// Address: 0x372d0c0
//
__int64 __fastcall sub_372D0C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rcx
  unsigned __int64 v13; // rcx
  __int64 *v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v17; // rdi
  __int64 v18; // r12
  __m128i v19; // [rsp+0h] [rbp-20h] BYREF

  v19.m128i_i64[0] = a2;
  v19.m128i_i64[1] = a3;
  v5 = sub_372CB80(a1, &v19);
  v8 = *(_QWORD *)v5;
  v9 = v5;
  v10 = *(_DWORD *)(v5 + 8);
  v11 = 16LL * v10;
  v12 = *(_QWORD *)(v8 + v11 - 16);
  if ( (v12 & 4) != 0 && a4 == (v12 & 0xFFFFFFFFFFFFFFF8LL) )
    return v10 - 1LL;
  v13 = *(unsigned int *)(v9 + 12);
  if ( v10 >= v13 )
  {
    v17 = v10 + 1LL;
    v18 = a4 | 4;
    if ( v13 < v17 )
    {
      sub_C8D5F0(v9, (const void *)(v9 + 16), v17, 0x10u, v6, v7);
      v8 = *(_QWORD *)v9;
      v11 = 16LL * *(unsigned int *)(v9 + 8);
    }
    *(_QWORD *)(v8 + v11) = v18;
    *(_QWORD *)(v8 + v11 + 8) = -1;
    v15 = (unsigned int)(*(_DWORD *)(v9 + 8) + 1);
    *(_DWORD *)(v9 + 8) = v15;
  }
  else
  {
    v14 = (__int64 *)(v11 + v8);
    if ( v14 )
    {
      v14[1] = -1;
      *v14 = a4 | 4;
      v10 = *(_DWORD *)(v9 + 8);
    }
    v15 = v10 + 1;
    *(_DWORD *)(v9 + 8) = v15;
  }
  return v15 - 1;
}

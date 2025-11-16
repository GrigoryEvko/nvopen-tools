// Function: sub_1F776B0
// Address: 0x1f776b0
//
__int64 *__fastcall sub_1F776B0(__int64 a1, __int64 *a2, char a3, __m128i a4, double a5, __m128i a6)
{
  unsigned int *v7; // rax
  __int64 v8; // r12
  __int64 *result; // rax
  __int64 v10; // r14
  __int64 *v11; // rax
  __int64 v12; // r14
  __int64 v13; // r8
  unsigned __int64 v14; // r15
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rdi
  char *v18; // rax
  bool v19; // zf
  char v20; // r9
  const void **v21; // rax
  __int64 v22; // rax
  char v23; // r12
  __int64 *v24; // r14
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // r15
  int v27; // eax
  __int128 v28; // rax
  const void **v29; // [rsp-60h] [rbp-60h]
  __int64 *v30; // [rsp-60h] [rbp-60h]
  unsigned int v31; // [rsp-58h] [rbp-58h] BYREF
  const void **v32; // [rsp-50h] [rbp-50h]
  __int64 v33; // [rsp-48h] [rbp-48h] BYREF
  int v34; // [rsp-40h] [rbp-40h]

  if ( a3 )
    return 0;
  v7 = *(unsigned int **)(a1 + 32);
  v8 = *(_QWORD *)v7;
  if ( *(_WORD *)(*(_QWORD *)v7 + 24LL) != 137 )
    return 0;
  v10 = v7[2];
  if ( !sub_1D18C00(*(_QWORD *)v7, 1, v7[2]) )
    return 0;
  if ( *(_BYTE *)(*(_QWORD *)(v8 + 40) + 16 * v10) != 2 )
    return 0;
  v11 = *(__int64 **)(v8 + 32);
  v12 = *v11;
  v13 = *v11;
  v14 = v11[1];
  v15 = *((_DWORD *)v11 + 2);
  v16 = v11[10];
  v17 = v11[5];
  v18 = *(char **)(a1 + 40);
  v19 = *(_DWORD *)(v16 + 84) == 18;
  v20 = *v18;
  v21 = (const void **)*((_QWORD *)v18 + 1);
  LOBYTE(v31) = v20;
  v32 = v21;
  if ( !v19 )
    return 0;
  v22 = *(_QWORD *)(v13 + 40) + 16LL * v15;
  v23 = *(_BYTE *)v22;
  v29 = *(const void ***)(v22 + 8);
  if ( !sub_1D188A0(v17) || v23 != (_BYTE)v31 || !v23 && v29 != v32 )
    return 0;
  v33 = *(_QWORD *)(a1 + 72);
  if ( v33 )
    sub_1F6CA20(&v33);
  v34 = *(_DWORD *)(a1 + 64);
  v24 = sub_1D3C080(a2, (__int64)&v33, v12, v14, v31, v32, a4, a5, a6);
  v26 = v25;
  if ( (_BYTE)v31 )
    v27 = sub_1F6C8D0(v31);
  else
    v27 = sub_1F58D40((__int64)&v31);
  *(_QWORD *)&v28 = sub_1D38BB0((__int64)a2, (unsigned int)(v27 - 1), (__int64)&v33, v31, v32, 0, a4, a5, a6, 0);
  result = sub_1D332F0(
             a2,
             (unsigned int)(*(_WORD *)(a1 + 24) != 142) + 123,
             (__int64)&v33,
             v31,
             v32,
             0,
             *(double *)a4.m128i_i64,
             a5,
             a6,
             (__int64)v24,
             v26,
             v28);
  if ( v33 )
  {
    v30 = result;
    sub_161E7C0((__int64)&v33, v33);
    return v30;
  }
  return result;
}

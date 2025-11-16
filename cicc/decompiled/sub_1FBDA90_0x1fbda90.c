// Function: sub_1FBDA90
// Address: 0x1fbda90
//
__int64 *__fastcall sub_1FBDA90(__int64 *a1, __int64 a2, __m128 a3, __m128i a4, __m128i a5)
{
  bool v5; // r14
  _BOOL4 v6; // r13d
  __int64 v9; // rax
  __int64 v10; // rsi
  int v11; // r11d
  __int64 v12; // rdi
  __int64 *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r8
  int v16; // r9d
  int v17; // esi
  __int64 v18; // r10
  __int64 v19; // rdx
  __int64 v20; // rdi
  unsigned __int64 v21; // rdx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // r13
  unsigned __int64 v25; // rcx
  __int64 *result; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp-30h] [rbp-A0h]
  unsigned __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h] BYREF
  int v31; // [rsp+18h] [rbp-58h]
  __int64 *v32; // [rsp+20h] [rbp-50h] BYREF
  int v33; // [rsp+28h] [rbp-48h]
  char v34; // [rsp+2Ch] [rbp-44h]
  __int64 v35; // [rsp+30h] [rbp-40h]

  v5 = 0;
  v6 = 1;
  v9 = *(_QWORD *)(a2 + 48);
  if ( v9 && !*(_QWORD *)(v9 + 32) )
  {
    v27 = *(_QWORD *)(v9 + 16);
    v5 = *(_WORD *)(v27 + 24) == 191;
    v6 = *(_WORD *)(v27 + 24) != 191;
  }
  v10 = *(_QWORD *)(a2 + 72);
  v30 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v30, v10, 2);
  v11 = *((_DWORD *)a1 + 4);
  v12 = *a1;
  v31 = *(_DWORD *)(a2 + 64);
  v13 = *(__int64 **)(a2 + 32);
  v14 = *v13;
  v15 = v13[1];
  v16 = *(_DWORD *)(v13[10] + 84);
  v17 = **(unsigned __int8 **)(a2 + 40);
  v18 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL);
  v33 = v11;
  v19 = v13[6];
  v28 = v13[5];
  v35 = v12;
  v20 = a1[1];
  v32 = a1;
  v34 = 0;
  v24 = sub_20ACAE0(v20, v17, v18, v14, v15, v16, v28, v19, v6, (__int64)&v32, (__int64)&v30);
  v25 = v21;
  if ( v30 )
  {
    v29 = v21;
    sub_161E7C0((__int64)&v30, v30);
    v25 = v29;
  }
  if ( !v24 )
    return 0;
  if ( !v5 || *(_WORD *)(v24 + 24) == 137 )
    return (__int64 *)v24;
  result = sub_1FBD240(a1, v24, v25, v25, v22, v23, a3, a4, a5);
  if ( (__int64 *)a2 == result )
    return 0;
  if ( !result )
    return (__int64 *)v24;
  return result;
}

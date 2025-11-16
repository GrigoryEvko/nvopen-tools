// Function: sub_3449820
// Address: 0x3449820
//
__int64 __fastcall sub_3449820(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  int *v7; // rax
  int v8; // ecx
  _QWORD ***v9; // rax
  unsigned int *v10; // rdi
  __int64 v11; // rax
  unsigned int *v12; // rdi
  _QWORD **v13; // rcx
  _DWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  _QWORD *v17; // r10
  __int64 *v18; // rax
  __int64 v19; // r12
  int v20; // r9d
  __int128 *v21; // rcx
  __int128 *v22; // r15
  __int64 v23; // rsi
  unsigned int *v24; // r14
  unsigned __int8 *v25; // r12
  int v26; // edx
  int v27; // r14d
  __int64 v28; // rax
  __m128i v29; // xmm0
  __int64 result; // rax
  __int64 *v31; // rax
  int v32; // [rsp+Ch] [rbp-B4h]
  __int128 *v33; // [rsp+10h] [rbp-B0h]
  _QWORD *v34; // [rsp+18h] [rbp-A8h]
  __int64 v35; // [rsp+40h] [rbp-80h] BYREF
  __int64 v36; // [rsp+48h] [rbp-78h]
  _QWORD v37[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD v38[2]; // [rsp+60h] [rbp-60h] BYREF
  _QWORD v39[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v40; // [rsp+80h] [rbp-40h] BYREF
  int v41; // [rsp+88h] [rbp-38h]

  v7 = *(int **)(a1 + 16);
  v35 = a4;
  v36 = a5;
  v8 = *v7;
  v9 = *(_QWORD ****)(a1 + 8);
  v37[0] = a2;
  v10 = *(unsigned int **)(a1 + 24);
  v37[1] = a3;
  v11 = sub_3449670(v10, a2, a3, *(_QWORD ***)a1, *v9, v8 + 1, a6);
  v12 = *(unsigned int **)(a1 + 24);
  v13 = *(_QWORD ***)a1;
  v38[0] = v11;
  v14 = *(_DWORD **)(a1 + 16);
  v38[1] = v15;
  v39[0] = sub_3449670(v12, v35, v36, v13, **(_QWORD ****)(a1 + 8), *v14 + 1, a6);
  v39[1] = v16;
  if ( v38[0] )
  {
    v17 = **(_QWORD ***)(a1 + 8);
    v18 = *(__int64 **)(a1 + 40);
    v19 = *v18;
    v20 = *(_DWORD *)(*v18 + 28);
    if ( v39[0] )
      v21 = (__int128 *)v39;
    else
      v21 = (__int128 *)&v35;
    v22 = (__int128 *)v38;
  }
  else
  {
    result = 0;
    if ( !v39[0] )
      return result;
    v21 = (__int128 *)v39;
    v22 = (__int128 *)v37;
    v17 = **(_QWORD ***)(a1 + 8);
    v31 = *(__int64 **)(a1 + 40);
    v19 = *v31;
    v20 = *(_DWORD *)(*v31 + 28);
  }
  v23 = *(_QWORD *)(v19 + 80);
  v24 = *(unsigned int **)(a1 + 48);
  v40 = v23;
  if ( v23 )
  {
    v32 = v20;
    v33 = v21;
    v34 = v17;
    sub_B96E90((__int64)&v40, v23, 1);
    v20 = v32;
    v21 = v33;
    v17 = v34;
  }
  v41 = *(_DWORD *)(v19 + 72);
  v25 = sub_3405C90(v17, **(_DWORD **)(a1 + 32), (__int64)&v40, *v24, *((_QWORD *)v24 + 1), v20, a6, *v22, *v21);
  v27 = v26;
  if ( v40 )
    sub_B91220((__int64)&v40, v40);
  v28 = *(_QWORD *)(a1 + 8);
  v29 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a1 + 40));
  *(_QWORD *)(v28 + 16) = v29.m128i_i64[0];
  *(_DWORD *)(v28 + 24) = v29.m128i_i32[2];
  *(_QWORD *)(v28 + 32) = v25;
  *(_DWORD *)(v28 + 40) = v27;
  return 1;
}

// Function: sub_1F7E7D0
// Address: 0x1f7e7d0
//
__int64 *__fastcall sub_1F7E7D0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        double a7,
        double a8,
        __m128i a9)
{
  int v11; // ebx
  __int64 *v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // r13
  __int64 v16; // r15
  __int64 *result; // rax
  __int64 v18; // rax
  int v19; // ecx
  int v20; // r8d
  int v21; // r9d
  __int64 v22; // rax
  char v23; // dl
  __int64 v24; // rax
  __m128i v25; // xmm0
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r9
  __int64 v30; // rax
  _QWORD *v31; // r13
  int v32; // eax
  __int64 v33; // r8
  __int128 v34; // rax
  __int128 v35; // rax
  unsigned int v36; // eax
  __int64 v37; // [rsp+8h] [rbp-88h]
  unsigned int v38; // [rsp+10h] [rbp-80h]
  __int128 v39; // [rsp+10h] [rbp-80h]
  __int64 *v40; // [rsp+10h] [rbp-80h]
  __int64 *v41; // [rsp+10h] [rbp-80h]
  unsigned int v42; // [rsp+20h] [rbp-70h] BYREF
  const void **v43; // [rsp+28h] [rbp-68h]
  __int64 v44; // [rsp+30h] [rbp-60h] BYREF
  int v45; // [rsp+38h] [rbp-58h]
  __int64 v46; // [rsp+40h] [rbp-50h] BYREF
  unsigned int v47; // [rsp+48h] [rbp-48h]
  __int64 v48; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v49; // [rsp+58h] [rbp-38h]

  v11 = *(unsigned __int16 *)(a1 + 24);
  v12 = *(__int64 **)(a1 + 32);
  if ( v11 == 52 )
  {
    v13 = v12[5];
    v14 = v12[6];
  }
  else
  {
    v13 = *v12;
    v14 = v12[1];
    v12 += 5;
  }
  v15 = *v12;
  v38 = *((_DWORD *)v12 + 2);
  v16 = sub_1D1ADA0(v13, v14, a3, a4, a5, a6);
  if ( !v16 )
    return 0;
  if ( *(_WORD *)(v15 + 24) != 124 )
    return 0;
  v18 = *(_QWORD *)(v15 + 32);
  v37 = *(_QWORD *)v18;
  if ( !sub_1D18C00(*(_QWORD *)v18, 1, *(_DWORD *)(v18 + 8)) )
    return 0;
  if ( *(_WORD *)(v37 + 24) != 120 )
    return 0;
  if ( !(unsigned __int8)sub_1F709E0(
                           *(_QWORD *)(*(_QWORD *)(v37 + 32) + 40LL),
                           *(_QWORD *)(*(_QWORD *)(v37 + 32) + 48LL)) )
    return 0;
  v22 = *(_QWORD *)(v15 + 40) + 16LL * v38;
  v23 = *(_BYTE *)v22;
  v43 = *(const void ***)(v22 + 8);
  v24 = *(_QWORD *)(v15 + 32);
  LOBYTE(v42) = v23;
  v25 = _mm_loadu_si128((const __m128i *)(v24 + 40));
  v26 = sub_1D1ADA0(v25.m128i_i64[0], v25.m128i_u32[2], v25.m128i_i64[1], v19, v20, v21);
  if ( !v26 )
    return 0;
  v30 = *(_QWORD *)(v26 + 88);
  v31 = *(_QWORD **)(v30 + 24);
  if ( *(_DWORD *)(v30 + 32) > 0x40u )
    v31 = (_QWORD *)*v31;
  v32 = sub_1D159C0((__int64)&v42, v25.m128i_i64[1], v27, v28, v37, v29);
  v33 = v37;
  if ( (_QWORD *)(unsigned int)(v32 - 1) != v31 )
    return 0;
  v44 = *(_QWORD *)(a1 + 72);
  if ( v44 )
  {
    sub_1F6CA20(&v44);
    v33 = v37;
  }
  v45 = *(_DWORD *)(a1 + 64);
  *(_QWORD *)&v34 = sub_1D332F0(
                      a2,
                      (unsigned int)(v11 != 52) + 123,
                      (__int64)&v44,
                      v42,
                      v43,
                      0,
                      *(double *)v25.m128i_i64,
                      a8,
                      a9,
                      **(_QWORD **)(v33 + 32),
                      *(_QWORD *)(*(_QWORD *)(v33 + 32) + 8LL),
                      *(_OWORD *)&v25);
  v39 = v34;
  sub_13A38D0((__int64)&v48, *(_QWORD *)(v16 + 88) + 24LL);
  if ( v11 == 52 )
  {
    sub_16A7490((__int64)&v48, 1);
    v36 = v49;
    v49 = 0;
    v47 = v36;
    v46 = v48;
    sub_135E100(&v48);
  }
  else
  {
    sub_16A7800((__int64)&v48, 1u);
    v47 = v49;
    v46 = v48;
  }
  *(_QWORD *)&v35 = sub_1D38970((__int64)a2, (__int64)&v46, (__int64)&v44, v42, v43, 0, v25, a8, a9, 0);
  result = sub_1D332F0(
             a2,
             52,
             (__int64)&v44,
             v42,
             v43,
             0,
             *(double *)v25.m128i_i64,
             a8,
             a9,
             v39,
             *((unsigned __int64 *)&v39 + 1),
             v35);
  if ( v47 > 0x40 && v46 )
  {
    v40 = result;
    j_j___libc_free_0_0(v46);
    result = v40;
  }
  if ( v44 )
  {
    v41 = result;
    sub_161E7C0((__int64)&v44, v44);
    return v41;
  }
  return result;
}

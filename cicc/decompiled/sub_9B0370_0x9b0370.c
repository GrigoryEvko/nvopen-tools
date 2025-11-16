// Function: sub_9B0370
// Address: 0x9b0370
//
__int64 __fastcall sub_9B0370(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 *a5, __int64 a6, __int128 a7)
{
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r13d
  __m128i v16; // xmm0
  bool v17; // al
  char v18; // r13
  unsigned int v19; // r15d
  __int64 v20; // rbx
  __int64 *v21; // rdx
  __int64 v22; // rbx
  unsigned int v23; // ebx
  __int64 v24; // r15
  __int64 v25; // r15
  unsigned int v26; // eax
  __int64 v27; // rdx
  unsigned int v28; // ecx
  __int64 v29; // rdx
  __int64 *v31; // rbx
  __int64 v33; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v34; // [rsp+28h] [rbp-C8h]
  __int64 v35; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-B8h]
  __int64 v37; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v38; // [rsp+48h] [rbp-A8h]
  __int64 v39; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v40; // [rsp+58h] [rbp-98h]
  int v41; // [rsp+60h] [rbp-90h] BYREF
  __int64 *v42; // [rsp+68h] [rbp-88h]
  __m128i v43; // [rsp+70h] [rbp-80h]
  __int64 v44; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v45; // [rsp+88h] [rbp-68h]
  __int64 v46; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v47; // [rsp+98h] [rbp-58h]
  __int64 v48; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v49; // [rsp+A8h] [rbp-48h]
  __int64 v50; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v51; // [rsp+B8h] [rbp-38h]

  v11 = *a5;
  v12 = *(_QWORD *)(a2 + 8);
  v34 = 1;
  v33 = 0;
  v36 = 1;
  v35 = 0;
  v48 = sub_9208B0(v11, v12);
  v49 = v13;
  v14 = sub_CA1930(&v48);
  sub_9B8920(v14, a3, &v33, &v35);
  v42 = a5;
  v15 = v36;
  v16 = _mm_loadu_si128((const __m128i *)&a7);
  v41 = a4;
  v43 = v16;
  if ( v36 <= 0x40 )
    v17 = v35 == 0;
  else
    v17 = v15 == (unsigned int)sub_C444A0(&v35);
  v18 = *(_BYTE *)(a2 + 7) & 0x40;
  if ( v17 )
  {
    if ( v18 )
      v31 = *(__int64 **)(a2 - 8);
    else
      v31 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    sub_9B01D0(a1, (__int64)&v41, *v31, (__int64)&v33);
    goto LABEL_35;
  }
  v19 = v34;
  if ( v34 <= 0x40 )
  {
    if ( !v33 )
      goto LABEL_6;
  }
  else if ( v19 == (unsigned int)sub_C444A0(&v33) )
  {
LABEL_6:
    if ( v18 )
      v20 = *(_QWORD *)(a2 - 8);
    else
      v20 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
    sub_9B01D0(a1, (__int64)&v41, *(_QWORD *)(v20 + 32), (__int64)&v35);
    goto LABEL_35;
  }
  if ( v18 )
    v21 = *(__int64 **)(a2 - 8);
  else
    v21 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  sub_9B01D0((__int64)&v44, (__int64)&v41, *v21, (__int64)&v33);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v22 = *(_QWORD *)(a2 - 8);
  else
    v22 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_9B01D0((__int64)&v48, (__int64)&v41, *(_QWORD *)(v22 + 32), (__int64)&v35);
  v23 = v47;
  v38 = v47;
  if ( v47 <= 0x40 )
  {
    v24 = v46;
LABEL_16:
    v25 = v50 & v24;
    v37 = v25;
    goto LABEL_17;
  }
  sub_C43780(&v37, &v46);
  v23 = v38;
  if ( v38 <= 0x40 )
  {
    v24 = v37;
    goto LABEL_16;
  }
  sub_C43B90(&v37, &v50);
  v23 = v38;
  v25 = v37;
LABEL_17:
  v38 = 0;
  v26 = v45;
  v40 = v45;
  if ( v45 <= 0x40 )
  {
    v27 = v44;
    v28 = 0;
LABEL_19:
    v29 = v48 & v27;
    goto LABEL_20;
  }
  sub_C43780(&v39, &v44);
  v26 = v40;
  if ( v40 <= 0x40 )
  {
    v27 = v39;
    v28 = v38;
    goto LABEL_19;
  }
  sub_C43B90(&v39, &v48);
  v26 = v40;
  v29 = v39;
  v28 = v38;
LABEL_20:
  *(_DWORD *)(a1 + 8) = v26;
  *(_QWORD *)a1 = v29;
  *(_DWORD *)(a1 + 24) = v23;
  *(_QWORD *)(a1 + 16) = v25;
  if ( v28 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v51 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
  if ( (unsigned int)v49 > 0x40 && v48 )
    j_j___libc_free_0_0(v48);
  if ( v47 > 0x40 && v46 )
    j_j___libc_free_0_0(v46);
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
LABEL_35:
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  return a1;
}

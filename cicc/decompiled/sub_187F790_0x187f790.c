// Function: sub_187F790
// Address: 0x187f790
//
__int64 __fastcall sub_187F790(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // rax
  _QWORD *v11; // r12
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r13
  __int128 *v15; // r15
  __int64 v16; // rbx
  void *v17; // r12
  __int64 v18; // r14
  unsigned __int64 v19; // r13
  size_t v20; // rdx
  int v21; // eax
  __int64 v22; // r13
  size_t v23; // r14
  int v24; // r14d
  __int64 v25; // rsi
  double v26; // xmm4_8
  double v27; // xmm5_8
  unsigned int v29; // ecx
  __int64 v30; // r9
  __int64 v31; // [rsp+8h] [rbp-F8h]
  __int64 v32; // [rsp+10h] [rbp-F0h]
  __int64 v33; // [rsp+18h] [rbp-E8h]
  __int64 v34; // [rsp+20h] [rbp-E0h]
  void *v35; // [rsp+28h] [rbp-D8h]
  __int64 v36; // [rsp+30h] [rbp-D0h]
  __int64 v37; // [rsp+48h] [rbp-B8h]
  _BYTE *v38; // [rsp+50h] [rbp-B0h]
  _BYTE *v39; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+68h] [rbp-98h]
  __int64 *v41[2]; // [rsp+70h] [rbp-90h] BYREF
  __int64 **v42[2]; // [rsp+80h] [rbp-80h] BYREF
  void *s2[2]; // [rsp+90h] [rbp-70h] BYREF
  __int128 v44; // [rsp+A0h] [rbp-60h] BYREF
  __int128 v45; // [rsp+B0h] [rbp-50h]
  __int64 v46; // [rsp+C0h] [rbp-40h]

  v10 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v10 + 16) != 19 )
    sub_16BD130("Second argument of llvm.type.test must be metadata", 1u);
  v38 = *(_BYTE **)(v10 + 24);
  if ( *v38 )
    sub_16BD130("Second argument of llvm.type.test must be a metadata string", 1u);
  v11 = (_QWORD *)a2;
  v39 = (_BYTE *)sub_161E970((__int64)v38);
  v40 = v12;
  v13 = *(_QWORD *)(a1 + 16);
  if ( v39 )
  {
    s2[0] = &v44;
    sub_18736F0((__int64 *)s2, v39, (__int64)&v39[v40]);
    v14 = *(_QWORD *)(v13 + 96);
    v15 = (__int128 *)s2[0];
    v37 = v13 + 88;
    if ( !v14 )
      goto LABEL_24;
  }
  else
  {
    v15 = &v44;
    s2[1] = 0;
    s2[0] = &v44;
    LOBYTE(v44) = 0;
    v14 = *(_QWORD *)(v13 + 96);
    v37 = v13 + 88;
    if ( !v14 )
    {
LABEL_26:
      v46 = 0;
      a3 = 0;
      *(_OWORD *)s2 = 0;
      v44 = 0;
      v45 = 0;
LABEL_27:
      v25 = sub_159C540(**(__int64 ***)a1);
      goto LABEL_23;
    }
  }
  v16 = v37;
  v17 = s2[1];
  v18 = v14;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v18 + 40);
      v20 = (size_t)v17;
      if ( v19 <= (unsigned __int64)v17 )
        v20 = *(_QWORD *)(v18 + 40);
      if ( v20 )
      {
        v21 = memcmp(*(const void **)(v18 + 32), v15, v20);
        if ( v21 )
          break;
      }
      v22 = v19 - (_QWORD)v17;
      if ( v22 >= 0x80000000LL )
        goto LABEL_15;
      if ( v22 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v21 = v22;
        break;
      }
LABEL_6:
      v18 = *(_QWORD *)(v18 + 24);
      if ( !v18 )
        goto LABEL_16;
    }
    if ( v21 < 0 )
      goto LABEL_6;
LABEL_15:
    v16 = v18;
    v18 = *(_QWORD *)(v18 + 16);
  }
  while ( v18 );
LABEL_16:
  v23 = (size_t)v17;
  v11 = (_QWORD *)a2;
  if ( v16 == v37 || sub_1872D20(v15, v23, *(const void **)(v16 + 32), *(_QWORD *)(v16 + 40)) < 0 )
  {
LABEL_24:
    if ( v15 != &v44 )
      j_j___libc_free_0(v15, v44 + 1);
    goto LABEL_26;
  }
  if ( v15 != &v44 )
    j_j___libc_free_0(v15, v44 + 1);
  v24 = *(_DWORD *)(v16 + 64);
  v41[1] = (__int64 *)&v39;
  v41[0] = (__int64 *)a1;
  v42[0] = (__int64 **)a1;
  v42[1] = v41;
  if ( v24 )
  {
    v35 = (void *)sub_1875AC0(v41, (__int64)"global_addr", 11);
    if ( ((unsigned int)(v24 - 1) <= 1 || v24 == 4)
      && (v34 = sub_1875BD0(v42, (__int64)"align", 5, *(_QWORD *)(v16 + 72), 8, *(_QWORD *)(a1 + 48)),
          v33 = sub_1875BD0(
                  v42,
                  (__int64)"size_m1",
                  7,
                  *(_QWORD *)(v16 + 80),
                  *(_DWORD *)(v16 + 68),
                  *(_QWORD *)(a1 + 96)),
          v24 == 1) )
    {
      v32 = sub_1875AC0(v41, (__int64)"byte_array", 10);
      v31 = sub_1875BD0(v42, (__int64)"bit_mask", 8, *(unsigned __int8 *)(v16 + 88), 8, *(_QWORD *)(a1 + 56));
    }
    else if ( v24 == 2 )
    {
      v29 = *(_DWORD *)(v16 + 68);
      if ( v29 <= 5 )
        v30 = *(_QWORD *)(a1 + 72);
      else
        v30 = *(_QWORD *)(a1 + 88);
      v36 = sub_1875BD0(v42, (__int64)"inline_bits", 11, *(_QWORD *)(v16 + 96), 1 << v29, v30);
    }
  }
  LODWORD(s2[0]) = v24;
  s2[1] = v35;
  *(_QWORD *)&v44 = v34;
  *((_QWORD *)&v44 + 1) = v33;
  *(_QWORD *)&v45 = v32;
  *((_QWORD *)&v45 + 1) = v31;
  v46 = v36;
  if ( !v24 )
    goto LABEL_27;
  v25 = sub_187EA30((__int64 *)a1, (__int64)v38, a2, (__int64)s2, *(double *)a3.m128_u64, a4, a5);
LABEL_23:
  sub_164D160((__int64)v11, v25, a3, a4, a5, a6, v26, v27, a9, a10);
  return sub_15F20C0(v11);
}

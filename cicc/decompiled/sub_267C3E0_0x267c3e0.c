// Function: sub_267C3E0
// Address: 0x267c3e0
//
void __fastcall sub_267C3E0(__int64 a1)
{
  __int64 v1; // rax
  __int64 *v2; // rbx
  __int64 v4; // r15
  __int64 *v5; // rax
  __int64 v6; // r13
  __int64 *v7; // r12
  __int64 v8; // rax
  char *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  __m128i v14; // xmm1
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  unsigned __int64 *v17; // r12
  unsigned __int64 *v18; // r15
  unsigned __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // [rsp+18h] [rbp-3E8h]
  _BYTE *v23[2]; // [rsp+20h] [rbp-3E0h] BYREF
  __int64 v24; // [rsp+30h] [rbp-3D0h] BYREF
  __int64 *v25; // [rsp+40h] [rbp-3C0h]
  __int64 v26; // [rsp+48h] [rbp-3B8h]
  __int64 v27; // [rsp+50h] [rbp-3B0h] BYREF
  __m128i v28; // [rsp+60h] [rbp-3A0h] BYREF
  __int64 *v29; // [rsp+70h] [rbp-390h] BYREF
  int v30; // [rsp+78h] [rbp-388h]
  char v31; // [rsp+7Ch] [rbp-384h]
  __int64 v32; // [rsp+80h] [rbp-380h] BYREF
  __m128i v33; // [rsp+88h] [rbp-378h] BYREF
  __int64 v34; // [rsp+98h] [rbp-368h]
  __m128i v35; // [rsp+A0h] [rbp-360h] BYREF
  __m128i v36; // [rsp+B0h] [rbp-350h]
  unsigned __int64 *v37; // [rsp+C0h] [rbp-340h] BYREF
  __int64 v38; // [rsp+C8h] [rbp-338h]
  _BYTE v39[324]; // [rsp+D0h] [rbp-330h] BYREF
  int v40; // [rsp+214h] [rbp-1ECh]
  __int64 v41; // [rsp+218h] [rbp-1E8h]
  void *v42; // [rsp+220h] [rbp-1E0h] BYREF
  int v43; // [rsp+228h] [rbp-1D8h]
  char v44; // [rsp+22Ch] [rbp-1D4h]
  __int64 v45; // [rsp+230h] [rbp-1D0h]
  __m128i v46; // [rsp+238h] [rbp-1C8h] BYREF
  __int64 v47; // [rsp+248h] [rbp-1B8h]
  __m128i v48; // [rsp+250h] [rbp-1B0h] BYREF
  __m128i v49; // [rsp+260h] [rbp-1A0h] BYREF
  _BYTE v50[8]; // [rsp+270h] [rbp-190h] BYREF
  int v51; // [rsp+278h] [rbp-188h]
  char v52; // [rsp+3C0h] [rbp-40h]
  int v53; // [rsp+3C4h] [rbp-3Ch]
  __int64 v54; // [rsp+3C8h] [rbp-38h]

  v1 = *(_QWORD *)(a1 + 40);
  v2 = *(__int64 **)v1;
  v22 = *(_QWORD *)v1 + 8LL * *(unsigned int *)(v1 + 8);
  if ( v22 != *(_QWORD *)v1 )
  {
    do
    {
      while ( 1 )
      {
        v4 = *v2;
        if ( (unsigned __int8)sub_26747F0(*v2) )
          break;
LABEL_23:
        if ( (__int64 *)v22 == ++v2 )
          return;
      }
      v5 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 56))(*(_QWORD *)(a1 + 64), v4);
      v6 = *v5;
      v7 = v5;
      v8 = sub_B2BE50(*v5);
      if ( sub_B6EA50(v8)
        || (v20 = sub_B2BE50(v6),
            v21 = sub_B6F970(v20),
            (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v21 + 48LL))(v21)) )
      {
        sub_B179F0((__int64)&v42, (__int64)"openmp-opt", (__int64)"OpenMPGPU", 9, v4);
        sub_B18290((__int64)&v42, "OpenMP GPU kernel ", 0x12u);
        v9 = (char *)sub_BD5D20(v4);
        sub_B16430((__int64)v23, "OpenMPGPUKernel", 0xFu, v9, v10);
        v29 = &v32;
        sub_266F100((__int64 *)&v29, v23[0], (__int64)&v23[0][(unsigned __int64)v23[1]]);
        v33.m128i_i64[1] = (__int64)&v35;
        sub_266F100(&v33.m128i_i64[1], v25, (__int64)v25 + v26);
        v36 = _mm_loadu_si128(&v28);
        sub_B180C0((__int64)&v42, (unsigned __int64)&v29);
        if ( (__m128i *)v33.m128i_i64[1] != &v35 )
          j_j___libc_free_0(v33.m128i_u64[1]);
        if ( v29 != &v32 )
          j_j___libc_free_0((unsigned __int64)v29);
        sub_B18290((__int64)&v42, "\n", 1u);
        v14 = _mm_loadu_si128(&v46);
        v15 = _mm_loadu_si128(&v48);
        v16 = _mm_loadu_si128(&v49);
        v37 = (unsigned __int64 *)v39;
        v30 = v43;
        v33 = v14;
        v31 = v44;
        v35 = v15;
        v32 = v45;
        v36 = v16;
        v29 = (__int64 *)&unk_49D9D40;
        v34 = v47;
        v38 = 0x400000000LL;
        if ( v51 )
          sub_26781A0((__int64)&v37, (__int64)v50, v11, v12, (__int64)v50, v13);
        v39[320] = v52;
        v40 = v53;
        v41 = v54;
        v29 = (__int64 *)&unk_49D9DE8;
        if ( v25 != &v27 )
          j_j___libc_free_0((unsigned __int64)v25);
        if ( (__int64 *)v23[0] != &v24 )
          j_j___libc_free_0((unsigned __int64)v23[0]);
        v42 = &unk_49D9D40;
        sub_23FD590((__int64)v50);
        sub_1049740(v7, (__int64)&v29);
        v17 = v37;
        v29 = (__int64 *)&unk_49D9D40;
        v18 = &v37[10 * (unsigned int)v38];
        if ( v37 != v18 )
        {
          do
          {
            v18 -= 10;
            v19 = v18[4];
            if ( (unsigned __int64 *)v19 != v18 + 6 )
              j_j___libc_free_0(v19);
            if ( (unsigned __int64 *)*v18 != v18 + 2 )
              j_j___libc_free_0(*v18);
          }
          while ( v17 != v18 );
          v18 = v37;
        }
        if ( v18 != (unsigned __int64 *)v39 )
          _libc_free((unsigned __int64)v18);
        goto LABEL_23;
      }
      ++v2;
    }
    while ( (__int64 *)v22 != v2 );
  }
}

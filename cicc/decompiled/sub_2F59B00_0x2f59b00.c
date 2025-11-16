// Function: sub_2F59B00
// Address: 0x2f59b00
//
__int64 __fastcall sub_2F59B00(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 *v6; // rax
  __int64 *v7; // rbx
  __int64 v8; // rdx
  float v9; // xmm0_4
  float v10; // xmm0_4
  float v11; // xmm0_4
  __int64 *v12; // rbx
  __int64 *v13; // r15
  __int64 v14; // rax
  __int64 v15; // r8
  int v16; // ecx
  __int64 v17; // rsi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // rdx
  float v23; // xmm0_4
  float v24; // xmm0_4
  float v25; // xmm0_4
  __int64 **v27; // r13
  __int64 v28; // rax
  __int64 v29; // rbx
  __m128i v30; // xmm1
  __int64 v31; // rax
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r13
  unsigned __int64 v34; // rdi
  int v35; // eax
  int v36; // r9d
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 *v39; // [rsp+8h] [rbp-208h]
  __int64 v40; // [rsp+18h] [rbp-1F8h] BYREF
  __m128i v41; // [rsp+20h] [rbp-1F0h] BYREF
  void *v42; // [rsp+30h] [rbp-1E0h] BYREF
  int v43; // [rsp+38h] [rbp-1D8h]
  int v44; // [rsp+3Ch] [rbp-1D4h]
  __int64 v45; // [rsp+40h] [rbp-1D0h]
  __m128i v46; // [rsp+48h] [rbp-1C8h]
  char *v47; // [rsp+58h] [rbp-1B8h]
  const char *v48; // [rsp+60h] [rbp-1B0h]
  __int64 v49; // [rsp+68h] [rbp-1A8h]
  char v50; // [rsp+78h] [rbp-198h]
  unsigned __int64 *v51; // [rsp+80h] [rbp-190h]
  __int64 v52; // [rsp+88h] [rbp-188h]
  _BYTE v53[324]; // [rsp+90h] [rbp-180h] BYREF
  int v54; // [rsp+1D4h] [rbp-3Ch]
  __int64 v55; // [rsp+1D8h] [rbp-38h]

  v6 = (__int64 *)a3[2];
  v7 = (__int64 *)a3[1];
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  v39 = v6;
  *(_OWORD *)a1 = 0;
  for ( *(_OWORD *)(a1 + 16) = 0; v39 != v7; *(float *)(a1 + 40) = *(float *)(a1 + 40) + *(float *)&v47 )
  {
    v8 = *v7++;
    sub_2F59B00(&v42, a2, v8);
    v9 = *(float *)(a1 + 24) + *(float *)v46.m128i_i32;
    *(_DWORD *)a1 += (_DWORD)v42;
    *(_DWORD *)(a1 + 4) += HIDWORD(v42);
    *(float *)(a1 + 24) = v9;
    v10 = *(float *)(a1 + 28) + *(float *)&v46.m128i_i32[1];
    *(_DWORD *)(a1 + 8) += v43;
    *(_DWORD *)(a1 + 12) += v44;
    *(float *)(a1 + 28) = v10;
    v11 = *(float *)(a1 + 32) + *(float *)&v46.m128i_i32[2];
    *(_DWORD *)(a1 + 16) += v45;
    *(_DWORD *)(a1 + 20) += HIDWORD(v45);
    *(float *)(a1 + 32) = v11;
    *(float *)(a1 + 36) = *(float *)(a1 + 36) + *(float *)&v46.m128i_i32[3];
  }
  v12 = (__int64 *)a3[4];
  v13 = (__int64 *)a3[5];
  if ( v12 != v13 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(a2 + 808);
      v15 = *v12;
      v16 = *(_DWORD *)(v14 + 24);
      v17 = *(_QWORD *)(v14 + 8);
      if ( !v16 )
        goto LABEL_5;
      v18 = v16 - 1;
      v19 = v18 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v15 != *v20 )
      {
        v35 = 1;
        while ( v21 != -4096 )
        {
          v36 = v35 + 1;
          v19 = v18 & (v35 + v19);
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( v15 == *v20 )
            goto LABEL_8;
          v35 = v36;
        }
        goto LABEL_5;
      }
LABEL_8:
      if ( (_QWORD *)v20[1] == a3 )
      {
        v22 = *v12++;
        sub_2F59030((__int64)&v42, a2, v22);
        v23 = *(float *)(a1 + 24) + *(float *)v46.m128i_i32;
        *(_DWORD *)a1 += (_DWORD)v42;
        *(_DWORD *)(a1 + 4) += HIDWORD(v42);
        *(float *)(a1 + 24) = v23;
        v24 = *(float *)(a1 + 28) + *(float *)&v46.m128i_i32[1];
        *(_DWORD *)(a1 + 8) += v43;
        *(_DWORD *)(a1 + 12) += v44;
        *(float *)(a1 + 28) = v24;
        v25 = *(float *)(a1 + 32) + *(float *)&v46.m128i_i32[2];
        *(_DWORD *)(a1 + 16) += v45;
        *(_DWORD *)(a1 + 20) += HIDWORD(v45);
        *(float *)(a1 + 32) = v25;
        *(float *)(a1 + 36) = *(float *)(a1 + 36) + *(float *)&v46.m128i_i32[3];
        *(float *)(a1 + 40) = *(float *)(a1 + 40) + *(float *)&v47;
        if ( v13 == v12 )
          break;
      }
      else
      {
LABEL_5:
        if ( v13 == ++v12 )
          break;
      }
    }
  }
  if ( *(_DWORD *)(a1 + 20)
     | *(_DWORD *)(a1 + 8)
     | *(_DWORD *)(a1 + 16)
     | *(_DWORD *)(a1 + 12)
     | *(_DWORD *)(a1 + 4)
     | *(_DWORD *)a1 )
  {
    v27 = *(__int64 ***)(a2 + 816);
    v28 = sub_B2BE50(**v27);
    if ( sub_B6EA50(v28)
      || (v37 = sub_B2BE50(**v27),
          v38 = sub_B6F970(v37),
          (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v38 + 48LL))(v38)) )
    {
      v29 = *(_QWORD *)a3[4];
      sub_2EA6600(&v40, (__int64)a3);
      sub_B157E0((__int64)&v41, &v40);
      v30 = _mm_loadu_si128(&v41);
      v31 = **(_QWORD **)(v29 + 32);
      LOBYTE(v44) = 2;
      v43 = 20;
      v45 = v31;
      v47 = "regalloc";
      v48 = "LoopSpillReloadCopies";
      v52 = 0x400000000LL;
      v49 = 21;
      v50 = 0;
      v51 = (unsigned __int64 *)v53;
      v53[320] = 0;
      v54 = -1;
      v55 = v29;
      v42 = &unk_4A27410;
      v46 = v30;
      if ( v40 )
        sub_B91220((__int64)&v40, v40);
      sub_2F551D0(a1, (__int64)&v42);
      sub_B18290((__int64)&v42, "generated in loop", 0x11u);
      sub_2EAFC50(v27, (__int64)&v42);
      v32 = v51;
      v42 = &unk_49D9D40;
      v33 = &v51[10 * (unsigned int)v52];
      if ( v51 != v33 )
      {
        do
        {
          v33 -= 10;
          v34 = v33[4];
          if ( (unsigned __int64 *)v34 != v33 + 6 )
            j_j___libc_free_0(v34);
          if ( (unsigned __int64 *)*v33 != v33 + 2 )
            j_j___libc_free_0(*v33);
        }
        while ( v32 != v33 );
        v33 = v51;
      }
      if ( v33 != (unsigned __int64 *)v53 )
        _libc_free((unsigned __int64)v33);
    }
  }
  return a1;
}

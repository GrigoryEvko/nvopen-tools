// Function: sub_26C8F20
// Address: 0x26c8f20
//
void __fastcall sub_26C8F20(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  _QWORD *v3; // rbx
  _QWORD *v5; // r12
  __int64 v6; // rdi
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // r15
  __int64 v10; // r15
  unsigned __int64 *v11; // r14
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rdi
  __m128i *v14; // r15
  _BOOL8 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // rdi
  _QWORD *v18; // r14
  int *v19; // rax
  size_t v20; // rdx
  __m128i *v21; // r14
  unsigned int v22; // esi
  unsigned __int8 *v23; // rax
  __int64 v24; // rcx
  unsigned __int8 **v25; // r11
  int v26; // r8d
  unsigned int v27; // r10d
  unsigned __int8 **v28; // r14
  unsigned __int8 *v29; // rdx
  __int128 v30; // rax
  __m128i v31; // rax
  _QWORD *v32; // r14
  _QWORD *v33; // rbx
  _QWORD *v34; // r14
  __m128i *v35; // r12
  __int64 v36; // rax
  __m128i v37; // xmm1
  unsigned __int64 v38; // r14
  __int64 v39; // rax
  _QWORD *v40; // rdx
  char v41; // di
  int v42; // ecx
  int v43; // ecx
  __int64 v44; // rdi
  char v45; // al
  __int64 v46; // [rsp+0h] [rbp-2D0h]
  _QWORD *v47; // [rsp+8h] [rbp-2C8h]
  __int64 v48; // [rsp+8h] [rbp-2C8h]
  _QWORD *v49; // [rsp+8h] [rbp-2C8h]
  __int64 v51; // [rsp+18h] [rbp-2B8h]
  _QWORD *v52; // [rsp+18h] [rbp-2B8h]
  __int64 v53; // [rsp+18h] [rbp-2B8h]
  _QWORD *v54; // [rsp+28h] [rbp-2A8h]
  unsigned __int8 *v55; // [rsp+38h] [rbp-298h] BYREF
  __m128i v56; // [rsp+40h] [rbp-290h] BYREF
  unsigned __int64 v57[2]; // [rsp+50h] [rbp-280h] BYREF
  __int64 v58; // [rsp+60h] [rbp-270h] BYREF
  __int64 *v59; // [rsp+70h] [rbp-260h]
  __int64 v60; // [rsp+80h] [rbp-250h] BYREF
  unsigned __int64 v61[2]; // [rsp+A0h] [rbp-230h] BYREF
  __int64 v62; // [rsp+B0h] [rbp-220h] BYREF
  __int64 *v63; // [rsp+C0h] [rbp-210h]
  __int64 v64; // [rsp+D0h] [rbp-200h] BYREF
  __m128i v65; // [rsp+F0h] [rbp-1E0h] BYREF
  __m128i v66; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v67; // [rsp+110h] [rbp-1C0h]
  unsigned __int64 *v68; // [rsp+140h] [rbp-190h]
  unsigned int v69; // [rsp+148h] [rbp-188h]
  char v70; // [rsp+150h] [rbp-180h] BYREF

  v3 = *(_QWORD **)(a2 + 32);
  v54 = &v3[2 * *(unsigned int *)(a2 + 40)];
  if ( v3 != v54 )
  {
    while ( 1 )
    {
      v5 = (_QWORD *)*v3;
      v6 = *(_QWORD *)(*v3 - 32LL);
      if ( !v6 )
        goto LABEL_3;
      if ( *(_BYTE *)v6 )
        goto LABEL_3;
      if ( v5[10] != *(_QWORD *)(v6 + 24) )
        goto LABEL_3;
      v55 = *(unsigned __int8 **)(*v3 - 32LL);
      if ( sub_B2FC80(v6) )
        goto LABEL_3;
      v7 = v5[5];
      v8 = *(__int64 **)(a1 + 1288);
      sub_B157E0((__int64)&v56, v5 + 6);
      sub_B17850((__int64)&v65, *(_QWORD *)(a1 + 1528), (__int64)"NotInline", 9, &v56, v7);
      sub_B18290((__int64)&v65, "previous inlining not repeated: '", 0x21u);
      sub_B16080((__int64)v57, "Callee", 6, v55);
      v9 = sub_B826F0((__int64)&v65, (__int64)v57);
      sub_B18290(v9, "' into '", 8u);
      sub_B16080((__int64)v61, "Caller", 6, a3);
      v10 = sub_B826F0(v9, (__int64)v61);
      sub_B18290(v10, "'", 1u);
      sub_1049740(v8, v10);
      if ( v63 != &v64 )
        j_j___libc_free_0((unsigned __int64)v63);
      if ( (__int64 *)v61[0] != &v62 )
        j_j___libc_free_0(v61[0]);
      if ( v59 != &v60 )
        j_j___libc_free_0((unsigned __int64)v59);
      if ( (__int64 *)v57[0] != &v58 )
        j_j___libc_free_0(v57[0]);
      v11 = v68;
      v65.m128i_i64[0] = (__int64)&unk_49D9D40;
      v12 = &v68[10 * v69];
      if ( v68 != v12 )
      {
        do
        {
          v12 -= 10;
          v13 = v12[4];
          if ( (unsigned __int64 *)v13 != v12 + 6 )
            j_j___libc_free_0(v13);
          if ( (unsigned __int64 *)*v12 != v12 + 2 )
            j_j___libc_free_0(*v12);
        }
        while ( v11 != v12 );
        v12 = v68;
      }
      if ( v12 != (unsigned __int64 *)&v70 )
        _libc_free((unsigned __int64)v12);
      v14 = (__m128i *)v3[1];
      if ( !v14[3].m128i_i64[1] && !sub_EF9210((_QWORD *)v3[1]) )
        goto LABEL_3;
      if ( (v14[3].m128i_i8[4] & 4) != 0 )
        goto LABEL_3;
      if ( !byte_4FF7C68 )
        break;
      if ( v14[4].m128i_i64[0] )
      {
LABEL_3:
        v3 += 2;
        if ( v54 == v3 )
          return;
      }
      else
      {
        v15 = sub_EF9210(v14);
        v16 = sub_C1B1E0(v15, 1u, 0, (bool *)v65.m128i_i8);
        v17 = v55;
        v14[4].m128i_i64[0] = v16;
        v18 = *(_QWORD **)(a1 + 1136);
        v19 = (int *)sub_26C07A0((__int64)v17);
        v21 = (__m128i *)sub_26C7880(v18, v19, v20);
        if ( !v21 )
        {
          *(_QWORD *)&v30 = sub_BD5D20((__int64)v55);
          v31.m128i_i64[0] = sub_C16140(v30, (__int64)"selected", 8);
          v32 = *(_QWORD **)(a1 + 1160);
          v66 = 0u;
          v65 = v31;
          v67 = 0;
          v52 = (_QWORD *)(a1 + 1152);
          if ( v32 )
          {
            v47 = v3;
            v33 = v32;
            v34 = (_QWORD *)(a1 + 1152);
            do
            {
              if ( sub_26BDDA0((__int64)(v33 + 4), (__int64)&v65) )
              {
                v33 = (_QWORD *)v33[3];
              }
              else
              {
                v34 = v33;
                v33 = (_QWORD *)v33[2];
              }
            }
            while ( v33 );
            v3 = v47;
            if ( v34 == v52 || (v35 = (__m128i *)(v34 + 9), sub_26BDDA0((__int64)&v65, (__int64)(v34 + 4))) )
            {
LABEL_45:
              v46 = (__int64)v34;
              v36 = sub_22077B0(0xF8u);
              v37 = _mm_loadu_si128(&v66);
              v38 = v36;
              *(__m128i *)(v36 + 32) = _mm_loadu_si128(&v65);
              v35 = (__m128i *)(v36 + 72);
              *(__m128i *)(v36 + 48) = v37;
              v48 = v36 + 32;
              *(_QWORD *)(v36 + 64) = v67;
              memset((void *)(v36 + 72), 0, 0xB0u);
              *(_QWORD *)(v36 + 168) = v36 + 152;
              *(_QWORD *)(v36 + 176) = v36 + 152;
              *(_QWORD *)(v36 + 216) = v36 + 200;
              *(_QWORD *)(v36 + 224) = v36 + 200;
              v39 = sub_26C8E00((_QWORD *)(a1 + 1144), v46, v36 + 32);
              if ( v40 )
              {
                if ( v52 == v40 || v39 )
                {
                  v41 = 1;
                }
                else
                {
                  v44 = v48;
                  v49 = v40;
                  v45 = sub_26BDDA0(v44, (__int64)(v40 + 4));
                  v40 = v49;
                  v41 = v45;
                }
                sub_220F040(v41, v38, v40, v52);
                ++*(_QWORD *)(a1 + 1184);
              }
              else
              {
                v53 = v39;
                sub_26BB480(0);
                j_j___libc_free_0(v38);
                v35 = (__m128i *)(v53 + 72);
              }
            }
            v21 = v35;
            goto LABEL_30;
          }
          v34 = (_QWORD *)(a1 + 1152);
          goto LABEL_45;
        }
LABEL_30:
        v3 += 2;
        sub_C1D5C0(v21, v14, 1u);
        sub_26BDE70((__int64)v21);
        if ( v54 == v3 )
          return;
      }
    }
    v22 = *(_DWORD *)(a1 + 1608);
    v51 = a1 + 1584;
    if ( v22 )
    {
      v23 = v55;
      v24 = *(_QWORD *)(a1 + 1592);
      v25 = 0;
      v26 = 1;
      v27 = (v22 - 1) & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
      v28 = (unsigned __int8 **)(v24 + 16LL * v27);
      v29 = *v28;
      if ( *v28 == v55 )
      {
LABEL_36:
        v28[1] += sub_EF9210(v14);
        goto LABEL_3;
      }
      while ( v29 != (unsigned __int8 *)-4096LL )
      {
        if ( !v25 && v29 == (unsigned __int8 *)-8192LL )
          v25 = v28;
        v27 = (v22 - 1) & (v26 + v27);
        v28 = (unsigned __int8 **)(v24 + 16LL * v27);
        v29 = *v28;
        if ( v55 == *v28 )
          goto LABEL_36;
        ++v26;
      }
      v43 = *(_DWORD *)(a1 + 1600);
      if ( v25 )
        v28 = v25;
      ++*(_QWORD *)(a1 + 1584);
      v42 = v43 + 1;
      v65.m128i_i64[0] = (__int64)v28;
      if ( 4 * v42 < 3 * v22 )
      {
        if ( v22 - *(_DWORD *)(a1 + 1604) - v42 > v22 >> 3 )
          goto LABEL_54;
        goto LABEL_53;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 1584);
      v65.m128i_i64[0] = 0;
    }
    v22 *= 2;
LABEL_53:
    sub_26BCF90(v51, v22);
    sub_26B9EB0(v51, (__int64 *)&v55, &v65);
    v23 = v55;
    v28 = (unsigned __int8 **)v65.m128i_i64[0];
    v42 = *(_DWORD *)(a1 + 1600) + 1;
LABEL_54:
    *(_DWORD *)(a1 + 1600) = v42;
    if ( *v28 != (unsigned __int8 *)-4096LL )
      --*(_DWORD *)(a1 + 1604);
    *v28 = v23;
    v28[1] = 0;
    goto LABEL_36;
  }
}

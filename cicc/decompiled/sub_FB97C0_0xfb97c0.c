// Function: sub_FB97C0
// Address: 0xfb97c0
//
__int64 __fastcall sub_FB97C0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v7; // ebx
  __int64 v8; // r14
  int v9; // ebx
  unsigned int v10; // r12d
  unsigned int v11; // r15d
  __int64 *v12; // rsi
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rbx
  __m128i *v21; // rax
  const __m128i *v22; // rdi
  __m128i *v23; // r12
  const __m128i *v24; // rdx
  __m128i *v25; // rcx
  __m128i *v26; // rdx
  __int64 *v27; // rbx
  __int64 *v28; // rcx
  const __m128i *v29; // r8
  __int64 v30; // rax
  int v31; // eax
  _QWORD *v32; // rdx
  _BYTE *v33; // rax
  __int64 v34; // rsi
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 v39; // r15
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // r9
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 v44; // r12
  _QWORD *v45; // rdi
  _BYTE *v46; // rdx
  _BYTE *v47; // rcx
  unsigned int v48; // eax
  __int64 v49; // r12
  _QWORD *v50; // rdi
  __int64 *v51; // [rsp+8h] [rbp-168h]
  __int64 *v53; // [rsp+20h] [rbp-150h]
  __int64 *v54; // [rsp+20h] [rbp-150h]
  const __m128i *v55; // [rsp+28h] [rbp-148h]
  __int64 v56; // [rsp+28h] [rbp-148h]
  __m128i v57; // [rsp+30h] [rbp-140h] BYREF
  const __m128i *v58; // [rsp+40h] [rbp-130h] BYREF
  const __m128i *v59; // [rsp+48h] [rbp-128h]
  __m128i *v60; // [rsp+50h] [rbp-120h]
  char v61; // [rsp+60h] [rbp-110h]
  __int64 v62; // [rsp+70h] [rbp-100h] BYREF
  __int64 *v63; // [rsp+78h] [rbp-F8h]
  __int64 v64; // [rsp+80h] [rbp-F0h]
  int v65; // [rsp+88h] [rbp-E8h]
  char v66; // [rsp+8Ch] [rbp-E4h]
  char v67; // [rsp+90h] [rbp-E0h] BYREF
  __int64 v68; // [rsp+D0h] [rbp-A0h] BYREF
  __int64 v69; // [rsp+D8h] [rbp-98h]
  __int64 v70; // [rsp+E0h] [rbp-90h]
  __int64 v71; // [rsp+E8h] [rbp-88h]
  __int64 *v72; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v73; // [rsp+F8h] [rbp-78h]
  _BYTE v74[112]; // [rsp+100h] [rbp-70h] BYREF

  v7 = *(_DWORD *)(a2 + 4);
  v63 = (__int64 *)&v67;
  v8 = *(_QWORD *)(a2 + 40);
  v72 = (__int64 *)v74;
  v62 = 0;
  v64 = 8;
  v65 = 0;
  v66 = 1;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v73 = 0x800000000LL;
  v9 = (v7 & 0x7FFFFFF) - 1;
  if ( !v9 )
  {
    if ( *(_QWORD *)(a1 + 8) )
    {
      v58 = 0;
      v11 = 0;
      v19 = 0;
      v15 = 0;
      v59 = 0;
      v60 = 0;
      goto LABEL_35;
    }
LABEL_64:
    v49 = sub_BD5C60(a2);
    v15 = unk_3F148B8;
    v50 = sub_BD2C40(72, unk_3F148B8);
    if ( v50 )
    {
      v15 = v49;
      sub_B4C8A0((__int64)v50, v49, a2 + 24, 0);
    }
    goto LABEL_59;
  }
  v10 = 0;
  v11 = 0;
  do
  {
    while ( 1 )
    {
      v16 = v10 + 1;
      v17 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32 * v16);
      v57.m128i_i64[0] = v17;
      if ( (*(_WORD *)(v17 + 2) & 0x7FFF) == 0 )
        goto LABEL_3;
      if ( v66 )
      {
        v18 = v63;
        a3 = &v63[HIDWORD(v64)];
        if ( v63 != a3 )
        {
          while ( v17 != *v18 )
          {
            if ( a3 == ++v18 )
              goto LABEL_14;
          }
          goto LABEL_12;
        }
LABEL_14:
        if ( HIDWORD(v64) < (unsigned int)v64 )
          break;
      }
      v15 = v17;
      sub_C8CC70((__int64)&v62, v17, (__int64)a3, v16, v17, a6);
      LODWORD(v16) = v10 + 1;
      if ( (_BYTE)a3 )
        goto LABEL_16;
LABEL_12:
      v13 = v57.m128i_i64[0];
      if ( (*(_WORD *)(v57.m128i_i64[0] + 2) & 0x7FFF) != 0 )
        goto LABEL_5;
LABEL_3:
      if ( (_DWORD)v70 )
      {
        sub_D6CB10((__int64)&v58, (__int64)&v68, v57.m128i_i64);
        if ( v61 )
        {
          v38 = (unsigned int)v73;
          v39 = v57.m128i_i64[0];
          v40 = (unsigned int)v73 + 1LL;
          if ( v40 > HIDWORD(v73) )
          {
            sub_C8D5F0((__int64)&v72, v74, v40, 8u, v36, v37);
            v38 = (unsigned int)v73;
          }
          v72[v38] = v39;
          v13 = v57.m128i_i64[0];
          LODWORD(v73) = v73 + 1;
          goto LABEL_5;
        }
LABEL_46:
        v13 = v57.m128i_i64[0];
        goto LABEL_5;
      }
      v12 = &v72[(unsigned int)v73];
      if ( v12 == sub_F8ED40(v72, (__int64)v12, v57.m128i_i64) )
      {
        v41 = v14 + 1;
        if ( v41 > HIDWORD(v73) )
        {
          v56 = v13;
          sub_C8D5F0((__int64)&v72, v74, v41, 8u, v13, v41);
          v13 = v56;
          v12 = &v72[(unsigned int)v73];
        }
        *v12 = v13;
        v42 = (unsigned int)(v73 + 1);
        LODWORD(v73) = v42;
        if ( (unsigned int)v42 > 8 )
        {
          v43 = v72;
          v51 = &v72[v42];
          do
          {
            v54 = v43;
            sub_D6CB10((__int64)&v58, (__int64)&v68, v43);
            v43 = v54 + 1;
          }
          while ( v51 != v54 + 1 );
          v13 = v57.m128i_i64[0];
          goto LABEL_5;
        }
        goto LABEL_46;
      }
LABEL_5:
      --v9;
      sub_AA5980(v13, v8, 0);
      v15 = v10;
      v11 = 1;
      sub_B54920(a2, v10);
      if ( v9 == v10 )
        goto LABEL_17;
    }
    v15 = (unsigned int)++HIDWORD(v64);
    *a3 = v17;
    ++v62;
LABEL_16:
    v10 = v16;
  }
  while ( v9 != (_DWORD)v16 );
LABEL_17:
  if ( *(_QWORD *)(a1 + 8) )
  {
    v58 = 0;
    v19 = (unsigned int)v73;
    v15 = 0;
    v59 = 0;
    v60 = 0;
    if ( (_DWORD)v73 )
    {
      v20 = (unsigned int)v73;
      v21 = (__m128i *)sub_22077B0(v20 * 16);
      v22 = v58;
      v23 = v21;
      v24 = v58;
      if ( v59 != v58 )
      {
        v25 = (__m128i *)((char *)v21 + (char *)v59 - (char *)v58);
        do
        {
          if ( v21 )
            *v21 = _mm_loadu_si128(v24);
          ++v21;
          ++v24;
        }
        while ( v25 != v21 );
      }
      if ( v22 )
        j_j___libc_free_0(v22, (char *)v60 - (char *)v22);
      v26 = &v23[v20];
      v27 = v72;
      v58 = v23;
      v59 = v23;
      v28 = &v72[(unsigned int)v73];
      v60 = v26;
      if ( v28 == v72 )
      {
        v15 = (__int64)v23;
        v19 = 0;
        goto LABEL_35;
      }
      v29 = &v57;
      while ( 1 )
      {
        v30 = *v27;
        v57.m128i_i64[0] = v8;
        v57.m128i_i64[1] = v30 | 4;
        if ( v23 == v26 )
        {
          v53 = v28;
          v55 = v29;
          ++v27;
          sub_F38BA0(&v58, v23, v29);
          v28 = v53;
          v23 = (__m128i *)v59;
          v29 = v55;
          if ( v53 == v27 )
            goto LABEL_34;
        }
        else
        {
          if ( v23 )
          {
            *v23 = _mm_loadu_si128(&v57);
            v23 = (__m128i *)v59;
          }
          ++v23;
          ++v27;
          v59 = v23;
          if ( v28 == v27 )
          {
LABEL_34:
            v15 = (__int64)v58;
            v19 = v23 - v58;
            break;
          }
        }
        v26 = v60;
      }
    }
LABEL_35:
    sub_FFB3D0(*(_QWORD *)(a1 + 8), v15, v19);
    if ( v58 )
    {
      v15 = (char *)v60 - (char *)v58;
      j_j___libc_free_0(v58, (char *)v60 - (char *)v58);
    }
  }
  v31 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v31 == 1 )
    goto LABEL_64;
  v32 = *(_QWORD **)(a2 - 8);
  if ( v31 == 2 )
  {
    v15 = 1;
    v44 = v32[4];
    v45 = sub_BD2C40(72, 1u);
    if ( v45 )
    {
      v15 = v44;
      sub_B4C8F0((__int64)v45, v44, 1u, a2 + 24, 0);
    }
LABEL_59:
    v11 = 1;
    sub_F91380((char *)a2);
    goto LABEL_40;
  }
  v33 = (_BYTE *)*v32;
  if ( *(_BYTE *)*v32 == 86 )
  {
    v46 = (_BYTE *)*((_QWORD *)v33 - 8);
    if ( *v46 == 4 )
    {
      v47 = (_BYTE *)*((_QWORD *)v33 - 4);
      if ( *v47 == 4 )
      {
        v15 = a2;
        v48 = sub_FB8CA0(a1, a2, *((_QWORD *)v33 - 12), *((_QWORD *)v46 - 4), *((_QWORD *)v47 - 4), 0, 0);
        if ( (_BYTE)v48 )
        {
          v11 = v48;
          *(_BYTE *)(a1 + 56) = 1;
        }
      }
    }
  }
LABEL_40:
  if ( v72 != (__int64 *)v74 )
    _libc_free(v72, v15);
  v34 = 8LL * (unsigned int)v71;
  sub_C7D6A0(v69, v34, 8);
  if ( !v66 )
    _libc_free(v63, v34);
  return v11;
}

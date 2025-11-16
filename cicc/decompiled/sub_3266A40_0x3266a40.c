// Function: sub_3266A40
// Address: 0x3266a40
//
__m128i *__fastcall sub_3266A40(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8)
{
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int8 v10; // al
  unsigned __int64 v11; // rax
  char v12; // r12
  unsigned __int16 *v13; // rdx
  unsigned __int16 v14; // ax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  __int64 v19; // r13
  const __m128i *v20; // rbx
  __int64 v21; // rax
  __m128i *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r14
  char v25; // r12
  unsigned __int16 *v26; // rdx
  unsigned __int16 v27; // ax
  __int64 v28; // rdx
  __int64 v29; // rax
  unsigned __int64 v30; // r12
  unsigned int v31; // eax
  unsigned __int64 v32; // r12
  unsigned int v33; // eax
  __m128i v34; // xmm7
  __int64 v35; // r15
  __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  char v38; // al
  unsigned __int64 v39; // rax
  char *v40; // rax
  unsigned __int64 v41; // r13
  unsigned __int16 *v42; // rdx
  unsigned __int16 v43; // ax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  __m128i *result; // rax
  const __m128i *v49; // rbx
  char v50; // r13
  unsigned __int64 v51; // r12
  unsigned __int16 *v52; // rdx
  unsigned __int16 v53; // ax
  __int64 v54; // rdx
  __int64 v55; // rax
  __m128i *v56; // rax
  __int64 v57; // rdx
  unsigned __int64 v58; // r13
  unsigned int v59; // eax
  __int64 v60; // r13
  __int64 v61; // r13
  unsigned int v62; // eax
  __m128i v63; // xmm5
  __int64 v65; // [rsp+10h] [rbp-100h]
  unsigned __int64 v66; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v67; // [rsp+28h] [rbp-E8h]
  unsigned int v68; // [rsp+28h] [rbp-E8h]
  unsigned int v69; // [rsp+28h] [rbp-E8h]
  char v70; // [rsp+28h] [rbp-E8h]
  unsigned int v71; // [rsp+28h] [rbp-E8h]
  unsigned int v72; // [rsp+28h] [rbp-E8h]
  __int64 v73; // [rsp+30h] [rbp-E0h]
  unsigned __int16 v74; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v75; // [rsp+48h] [rbp-C8h]
  __int64 v76; // [rsp+50h] [rbp-C0h]
  __int64 v77; // [rsp+58h] [rbp-B8h]
  unsigned __int16 v78; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+68h] [rbp-A8h]
  __int64 v80; // [rsp+70h] [rbp-A0h]
  __int64 v81; // [rsp+78h] [rbp-98h]
  unsigned __int16 v82; // [rsp+80h] [rbp-90h] BYREF
  __int64 v83; // [rsp+88h] [rbp-88h]
  __int64 v84; // [rsp+90h] [rbp-80h]
  __int64 v85; // [rsp+98h] [rbp-78h]
  __int64 v86; // [rsp+A0h] [rbp-70h]
  __int64 v87; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v88; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v89; // [rsp+B8h] [rbp-58h]
  __m128i v90; // [rsp+C0h] [rbp-50h] BYREF
  __m128i v91; // [rsp+D0h] [rbp-40h] BYREF

  v65 = (a3 - 1) / 2;
  if ( a2 >= v65 )
  {
    if ( (a3 & 1) != 0 )
    {
      v90 = _mm_loadu_si128((const __m128i *)&a7);
      v91 = _mm_loadu_si128((const __m128i *)&a8);
      result = (__m128i *)(a1 + 32 * a2);
      goto LABEL_61;
    }
    v73 = a2;
  }
  else
  {
    v73 = a2;
    do
    {
      v19 = 2 * (v73 + 1);
      v23 = (v73 + 1) << 6;
      v24 = a1 + v23 - 32;
      v20 = (const __m128i *)(a1 + v23);
      v25 = *(_BYTE *)sub_2E79000(*(__int64 **)(v20[1].m128i_i64[1] + 40));
      v66 = (unsigned __int32)v20[1].m128i_i32[0] >> 3;
      v26 = *(unsigned __int16 **)(v20->m128i_i64[1] + 48);
      v27 = *v26;
      v28 = *((_QWORD *)v26 + 1);
      v74 = v27;
      v75 = v28;
      if ( v27 )
      {
        if ( v27 == 1 || (unsigned __int16)(v27 - 504) <= 7u )
LABEL_64:
          BUG();
        v29 = 16LL * (v27 - 1);
        v9 = *(_QWORD *)&byte_444C4A0[v29];
        v10 = byte_444C4A0[v29 + 8];
      }
      else
      {
        v76 = sub_3007260((__int64)&v74);
        v77 = v8;
        v9 = v76;
        v10 = v77;
      }
      v90.m128i_i64[0] = v9;
      v90.m128i_i8[8] = v10;
      v11 = sub_CA1930(&v90);
      if ( v25 )
      {
        v32 = (unsigned int)(v11 >> 3) - v66;
        sub_3266230((__int64)&v90, (__int64)v20);
        if ( v90.m128i_i32[2] > 0x40u )
        {
          v33 = sub_C44630((__int64)&v90);
          if ( v90.m128i_i64[0] )
          {
            v69 = v33;
            j_j___libc_free_0_0(v90.m128i_u64[0]);
            v33 = v69;
          }
        }
        else
        {
          v33 = sub_39FAC40(v90.m128i_i64[0]);
        }
        v66 = v32 - (v33 >> 3);
      }
      v12 = *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(v24 + 24) + 40LL));
      v67 = *(_DWORD *)(v24 + 16) >> 3;
      v13 = *(unsigned __int16 **)(*(_QWORD *)(v24 + 8) + 48LL);
      v14 = *v13;
      v15 = *((_QWORD *)v13 + 1);
      v78 = v14;
      v79 = v15;
      if ( v14 )
      {
        if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
          goto LABEL_64;
        v17 = 16LL * (v14 - 1);
        v16 = *(_QWORD *)&byte_444C4A0[v17];
        LOBYTE(v17) = byte_444C4A0[v17 + 8];
      }
      else
      {
        v16 = sub_3007260((__int64)&v78);
        v80 = v16;
        v81 = v17;
      }
      v90.m128i_i64[0] = v16;
      v90.m128i_i8[8] = v17;
      v18 = sub_CA1930(&v90);
      if ( v12 )
      {
        v30 = (unsigned int)(v18 >> 3) - v67;
        sub_3266230((__int64)&v90, v24);
        if ( v90.m128i_i32[2] > 0x40u )
        {
          v31 = sub_C44630((__int64)&v90);
          if ( v90.m128i_i64[0] )
          {
            v68 = v31;
            j_j___libc_free_0_0(v90.m128i_u64[0]);
            v31 = v68;
          }
        }
        else
        {
          v31 = sub_39FAC40(v90.m128i_i64[0]);
        }
        v67 = v30 - (v31 >> 3);
      }
      if ( v67 > v66 )
      {
        --v19;
        v20 = (const __m128i *)(a1 + 32 * v19);
      }
      v21 = v73;
      v73 = v19;
      v22 = (__m128i *)(a1 + 32 * v21);
      *v22 = _mm_loadu_si128(v20);
      v22[1] = _mm_loadu_si128(v20 + 1);
    }
    while ( v19 < v65 );
    if ( (a3 & 1) != 0 )
      goto LABEL_29;
  }
  if ( (a3 - 2) / 2 == v73 )
  {
    v56 = (__m128i *)(a1 + 32 * v73);
    v57 = a1 + ((v73 + 1) << 6);
    *v56 = _mm_loadu_si128((const __m128i *)(v57 - 32));
    v56[1] = _mm_loadu_si128((const __m128i *)(v57 - 16));
    v73 = 2 * (v73 + 1) - 1;
  }
LABEL_29:
  v34 = _mm_loadu_si128((const __m128i *)&a8);
  v90 = _mm_loadu_si128((const __m128i *)&a7);
  v91 = v34;
  v35 = (v73 - 1) / 2;
  if ( v73 <= a2 )
  {
    result = (__m128i *)(a1 + 32 * v73);
  }
  else
  {
    while ( 1 )
    {
      v49 = (const __m128i *)(a1 + 32 * v35);
      v50 = *(_BYTE *)sub_2E79000(*(__int64 **)(v49[1].m128i_i64[1] + 40));
      v51 = (unsigned __int32)v49[1].m128i_i32[0] >> 3;
      v52 = *(unsigned __int16 **)(v49->m128i_i64[1] + 48);
      v53 = *v52;
      v54 = *((_QWORD *)v52 + 1);
      v82 = v53;
      v83 = v54;
      if ( v53 )
      {
        if ( v53 == 1 || (unsigned __int16)(v53 - 504) <= 7u )
          goto LABEL_64;
        v55 = 16LL * (v53 - 1);
        v37 = *(_QWORD *)&byte_444C4A0[v55];
        v38 = byte_444C4A0[v55 + 8];
      }
      else
      {
        v84 = sub_3007260((__int64)&v82);
        v85 = v36;
        v37 = v84;
        v38 = v85;
      }
      v88 = v37;
      LOBYTE(v89) = v38;
      v39 = sub_CA1930(&v88);
      if ( v50 )
      {
        v60 = (unsigned int)(v39 >> 3);
        sub_3266230((__int64)&v88, (__int64)v49);
        v61 = v60 - v51;
        if ( (unsigned int)v89 > 0x40 )
        {
          v62 = sub_C44630((__int64)&v88);
          if ( v88 )
          {
            v72 = v62;
            j_j___libc_free_0_0(v88);
            v62 = v72;
          }
        }
        else
        {
          v62 = sub_39FAC40(v88);
        }
        v51 = v61 - (v62 >> 3);
      }
      v40 = (char *)sub_2E79000(*(__int64 **)(v91.m128i_i64[1] + 40));
      v41 = (unsigned __int32)v91.m128i_i32[0] >> 3;
      v70 = *v40;
      v42 = *(unsigned __int16 **)(v90.m128i_i64[1] + 48);
      v43 = *v42;
      v44 = *((_QWORD *)v42 + 1);
      LOWORD(v88) = v43;
      v89 = v44;
      if ( v43 )
      {
        if ( v43 == 1 || (unsigned __int16)(v43 - 504) <= 7u )
          goto LABEL_64;
        v46 = 16LL * (v43 - 1);
        v45 = *(_QWORD *)&byte_444C4A0[v46];
        LOBYTE(v46) = byte_444C4A0[v46 + 8];
      }
      else
      {
        v45 = sub_3007260((__int64)&v88);
        v86 = v45;
        v87 = v46;
      }
      v88 = v45;
      LOBYTE(v89) = v46;
      v47 = sub_CA1930(&v88);
      if ( v70 )
      {
        v58 = (unsigned int)(v47 >> 3) - v41;
        sub_3266230((__int64)&v88, (__int64)&v90);
        if ( (unsigned int)v89 > 0x40 )
        {
          v59 = sub_C44630((__int64)&v88);
          if ( v88 )
          {
            v71 = v59;
            j_j___libc_free_0_0(v88);
            v59 = v71;
          }
        }
        else
        {
          v59 = sub_39FAC40(v88);
        }
        v41 = v58 - (v59 >> 3);
      }
      result = (__m128i *)(a1 + 32 * v73);
      if ( v41 <= v51 )
        break;
      v73 = v35;
      *result = _mm_loadu_si128(v49);
      result[1] = _mm_loadu_si128(v49 + 1);
      if ( a2 >= v35 )
      {
        result = (__m128i *)(a1 + 32 * v35);
        break;
      }
      v35 = (v35 - 1) / 2;
    }
  }
LABEL_61:
  v63 = _mm_loadu_si128(&v91);
  *result = _mm_loadu_si128(&v90);
  result[1] = v63;
  return result;
}

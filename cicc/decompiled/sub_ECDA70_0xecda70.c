// Function: sub_ECDA70
// Address: 0xecda70
//
__int64 __fastcall sub_ECDA70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  const __m128i *v6; // r14
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  unsigned __int64 v12; // rcx
  const void **v13; // rsi
  __int64 v14; // rcx
  int v15; // eax
  __m128i *v16; // rbx
  __int64 v17; // rax
  __m128i *v18; // rdi
  unsigned __int64 v19; // r15
  __int64 v21; // rax
  _DWORD *v22; // rcx
  unsigned __int64 v23; // rdx
  __int64 v24; // r14
  _DWORD *v25; // rbx
  int v26; // eax
  unsigned __int64 v27; // r12
  __m128i v28; // xmm0
  bool v29; // cc
  __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rax
  _DWORD *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rsi
  unsigned __int64 v39; // r15
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned __int64 v42; // rbx
  unsigned __int64 v43; // r9
  __int64 v44; // r12
  unsigned __int64 v45; // rdi
  __m128i v46; // xmm4
  int v47; // eax
  unsigned __int64 v48; // rax
  int v49; // edx
  __int64 v50; // rdi
  __int64 v51; // rdx
  size_t v52; // rdx
  unsigned __int64 v53; // rbx
  __int64 v54; // rdi
  __m128i v55; // xmm5
  unsigned int v56; // eax
  char *v57; // rbx
  char *v58; // [rsp+8h] [rbp-E8h]
  unsigned __int64 v59; // [rsp+10h] [rbp-E0h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  int v61; // [rsp+20h] [rbp-D0h] BYREF
  __m128i v62; // [rsp+28h] [rbp-C8h] BYREF
  const void *v63; // [rsp+38h] [rbp-B8h] BYREF
  unsigned int v64; // [rsp+40h] [rbp-B0h]
  __int64 v65; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD v66[3]; // [rsp+58h] [rbp-98h] BYREF
  _BYTE v67[64]; // [rsp+70h] [rbp-80h] BYREF
  __int64 v68; // [rsp+B0h] [rbp-40h]
  __int64 v69; // [rsp+B8h] [rbp-38h]

  v6 = (const __m128i *)&v65;
  v65 = a2;
  v66[0] = v67;
  v66[1] = 0;
  v66[2] = 64;
  v68 = 0;
  v69 = 0;
  sub_CA0EC0(a3, (__int64)v66);
  v11 = *(unsigned int *)(a1 + 24);
  v12 = *(unsigned int *)(a1 + 28);
  v68 = a4;
  v69 = a5;
  v13 = (const void **)(v11 + 1);
  if ( v11 + 1 > v12 )
  {
    v53 = *(_QWORD *)(a1 + 16);
    v54 = a1 + 16;
    if ( v53 > (unsigned __int64)&v65 || (unsigned __int64)&v65 >= v53 + 112 * v11 )
    {
      sub_ECD890(v54, (unsigned __int64)v13, v11, v12, v9, v10);
      v11 = *(unsigned int *)(a1 + 24);
      v14 = *(_QWORD *)(a1 + 16);
      v15 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      sub_ECD890(v54, (unsigned __int64)v13, v11, v12, v9, v10);
      v14 = *(_QWORD *)(a1 + 16);
      v11 = *(unsigned int *)(a1 + 24);
      v6 = (const __m128i *)((char *)&v66[-1] + v14 - v53);
      v15 = *(_DWORD *)(a1 + 24);
    }
  }
  else
  {
    v14 = *(_QWORD *)(a1 + 16);
    v15 = v11;
  }
  v16 = (__m128i *)(v14 + 112 * v11);
  if ( v16 )
  {
    v17 = v6->m128i_i64[0];
    v18 = v16 + 2;
    v16[1].m128i_i64[0] = 0;
    v16->m128i_i64[1] = (__int64)v16[2].m128i_i64;
    v16->m128i_i64[0] = v17;
    v16[1].m128i_i64[1] = 64;
    v19 = v6[1].m128i_u64[0];
    if ( v19 && &v16->m128i_u64[1] != &v6->m128i_u64[1] )
    {
      v52 = v6[1].m128i_u64[0];
      if ( v19 <= 0x40
        || (v13 = (const void **)&v16[2],
            sub_C8D290((__int64)&v16->m128i_i64[1], &v16[2], v52, 1u, (__int64)&v16->m128i_i64[1], v10),
            v52 = v6[1].m128i_u64[0],
            v18 = (__m128i *)v16->m128i_i64[1],
            v52) )
      {
        v13 = (const void **)v6->m128i_i64[1];
        memcpy(v18, v13, v52);
      }
      v16[1].m128i_i64[0] = v19;
    }
    v16[6] = _mm_loadu_si128(v6 + 6);
    v15 = *(_DWORD *)(a1 + 24);
  }
  *(_DWORD *)(a1 + 24) = v15 + 1;
  if ( *(_DWORD *)sub_ECD7B0(a1) == 1 )
  {
    v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 40LL))(a1);
    v13 = (const void **)0xCCCCCCCCCCCCCCCDLL;
    v22 = *(_DWORD **)(v21 + 8);
    v23 = *(unsigned int *)(v21 + 16);
    v24 = v21;
    v25 = v22 + 10;
    *(_BYTE *)(v21 + 115) = *v22 == 9;
    v26 = v23;
    v23 *= 40LL;
    v27 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v23 - 40) >> 3);
    if ( v23 > 0x28 )
    {
      do
      {
        v28 = _mm_loadu_si128((const __m128i *)(v25 + 2));
        v29 = *(v25 - 2) <= 0x40u;
        *(v25 - 10) = *v25;
        *((__m128i *)v25 - 2) = v28;
        if ( !v29 )
        {
          v30 = *((_QWORD *)v25 - 2);
          if ( v30 )
            j_j___libc_free_0_0(v30);
        }
        v31 = *((_QWORD *)v25 + 3);
        v25 += 10;
        *((_QWORD *)v25 - 7) = v31;
        LODWORD(v31) = *(v25 - 2);
        *(v25 - 2) = 0;
        *(v25 - 12) = v31;
        --v27;
      }
      while ( v27 );
      v26 = *(_DWORD *)(v24 + 16);
      v22 = *(_DWORD **)(v24 + 8);
    }
    v32 = (unsigned int)(v26 - 1);
    *(_DWORD *)(v24 + 16) = v32;
    v33 = &v22[10 * v32];
    if ( v33[8] > 0x40u )
    {
      v34 = *((_QWORD *)v33 + 3);
      if ( v34 )
        j_j___libc_free_0_0(v34);
    }
    if ( !*(_DWORD *)(v24 + 16) )
    {
      v60 = v24 + 8;
      (**(void (__fastcall ***)(int *, __int64))v24)(&v61, v24);
      v38 = *(unsigned int *)(v24 + 16);
      v39 = *(_QWORD *)(v24 + 8);
      v40 = v38;
      v41 = 40 * v38;
      v42 = v39 + 40 * v38;
      if ( v39 == v42 )
      {
        v13 = (const void **)(v38 + 1);
        if ( (unsigned __int64)v13 > *(unsigned int *)(v24 + 20) )
        {
          sub_EA9FB0(v60, (unsigned __int64)v13, v40, v35, v36, v37);
          LODWORD(v40) = *(_DWORD *)(v24 + 16);
          v42 = *(_QWORD *)(v24 + 8) + 40LL * (unsigned int)v40;
        }
        if ( v42 )
        {
          v55 = _mm_loadu_si128(&v62);
          *(_DWORD *)v42 = v61;
          *(__m128i *)(v42 + 8) = v55;
          v56 = v64;
          *(_DWORD *)(v42 + 32) = v64;
          if ( v56 > 0x40 )
          {
            v13 = &v63;
            sub_C43780(v42 + 24, &v63);
          }
          else
          {
            *(_QWORD *)(v42 + 24) = v63;
          }
          LODWORD(v40) = *(_DWORD *)(v24 + 16);
        }
        *(_DWORD *)(v24 + 16) = v40 + 1;
      }
      else
      {
        v43 = v38 + 1;
        v58 = (char *)&v61;
        v13 = (const void **)&v61;
        if ( v43 > *(unsigned int *)(v24 + 20) )
        {
          if ( v39 > (unsigned __int64)&v61 || v42 <= (unsigned __int64)&v61 )
          {
            sub_EA9FB0(v60, v43, v40, v35, v36, v43);
            v39 = *(_QWORD *)(v24 + 8);
            v13 = (const void **)&v61;
            v40 = *(unsigned int *)(v24 + 16);
            v41 = 40 * v40;
            v42 = v39 + 40 * v40;
          }
          else
          {
            v57 = (char *)&v61 - v39;
            sub_EA9FB0(v60, v43, v40, v35, v36, v43);
            v39 = *(_QWORD *)(v24 + 8);
            v40 = *(unsigned int *)(v24 + 16);
            v13 = (const void **)&v57[v39];
            v41 = 40 * v40;
            v58 = &v57[v39];
            v42 = v39 + 40 * v40;
          }
        }
        v44 = v39 + v41 - 40;
        v45 = v39;
        if ( v42 )
        {
          v46 = _mm_loadu_si128((const __m128i *)(v44 + 8));
          *(_DWORD *)v42 = *(_DWORD *)v44;
          *(__m128i *)(v42 + 8) = v46;
          v47 = *(_DWORD *)(v44 + 32);
          *(_DWORD *)(v44 + 32) = 0;
          *(_DWORD *)(v42 + 32) = v47;
          *(_QWORD *)(v42 + 24) = *(_QWORD *)(v44 + 24);
          v45 = *(_QWORD *)(v24 + 8);
          v40 = *(unsigned int *)(v24 + 16);
          v42 = v45 + 40 * v40;
          v44 = v42 - 40;
        }
        v48 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)(v44 - v39) >> 3);
        if ( (__int64)(v44 - v39) > 0 )
        {
          do
          {
            v49 = *(_DWORD *)(v44 - 40);
            v42 -= 40LL;
            v44 -= 40;
            v29 = *(_DWORD *)(v42 + 32) <= 0x40u;
            *(_DWORD *)v42 = v49;
            *(__m128i *)(v42 + 8) = _mm_loadu_si128((const __m128i *)(v44 + 8));
            if ( !v29 )
            {
              v50 = *(_QWORD *)(v42 + 24);
              if ( v50 )
              {
                v59 = v48;
                j_j___libc_free_0_0(v50);
                v48 = v59;
              }
            }
            *(_QWORD *)(v42 + 24) = *(_QWORD *)(v44 + 24);
            *(_DWORD *)(v42 + 32) = *(_DWORD *)(v44 + 32);
            *(_DWORD *)(v44 + 32) = 0;
            --v48;
          }
          while ( v48 );
          LODWORD(v40) = *(_DWORD *)(v24 + 16);
          v45 = *(_QWORD *)(v24 + 8);
        }
        v51 = (unsigned int)(v40 + 1);
        *(_DWORD *)(v24 + 16) = v51;
        if ( v39 <= (unsigned __int64)v58 && v45 + 40 * v51 > (unsigned __int64)v58 )
          v13 += 5;
        v29 = *(_DWORD *)(v39 + 32) <= 0x40u;
        *(_DWORD *)v39 = *(_DWORD *)v13;
        *(__m128i *)(v39 + 8) = _mm_loadu_si128((const __m128i *)(v13 + 1));
        if ( v29 && *((_DWORD *)v13 + 8) <= 0x40u )
        {
          *(_QWORD *)(v39 + 24) = v13[3];
          *(_DWORD *)(v39 + 32) = *((_DWORD *)v13 + 8);
        }
        else
        {
          v13 += 3;
          sub_C43990(v39 + 24, (__int64)v13);
        }
      }
      if ( v64 > 0x40 && v63 )
        j_j___libc_free_0_0(v63);
    }
  }
  if ( (_BYTE *)v66[0] != v67 )
    _libc_free(v66[0], v13);
  return 1;
}

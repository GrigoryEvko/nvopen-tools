// Function: sub_3990FF0
// Address: 0x3990ff0
//
void __fastcall sub_3990FF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rax
  __m128i *v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  const __m128i *v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  const __m128i *i; // r14
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 *v17; // r14
  __int64 v18; // rax
  char *v19; // rdx
  char v20; // si
  __int32 v21; // esi
  bool v22; // zf
  __m128i v23; // xmm1
  __m128i v24; // xmm0
  int v25; // r8d
  int v26; // r9d
  unsigned __int64 v27; // rax
  __int64 v28; // rcx
  unsigned int v29; // eax
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 *v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rax
  int v35; // r14d
  __m128i *v36; // r14
  __int64 v37; // r12
  __int64 v38; // r12
  __int64 v39; // r12
  __int64 v40; // rdx
  __m128i v41; // xmm5
  __m128i *v42; // rax
  __int64 v43; // rdx
  __int64 v44; // rdi
  int v45; // ebx
  void *v46; // r12
  size_t v47; // r15
  __int64 v48; // rcx
  __m128i *v49; // r15
  __int64 v50; // rbx
  __m128i *v51; // r12
  __m128i *v52; // rsi
  unsigned __int64 v53; // rax
  __m128i *v54; // rbx
  const __m128i *v55; // rdi
  signed __int64 v56; // rbx
  __m128i *v57; // rax
  _BYTE *v58; // rdx
  __m128i *v59; // rcx
  __m128i *v60; // rax
  __int64 v61; // rbx
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // rdi
  int v66; // esi
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int64 v69; // rdi
  __int64 v70; // r12
  unsigned int v71; // r14d
  __int64 v72; // r12
  unsigned int v73; // r14d
  __int64 v74; // r12
  unsigned int v75; // r14d
  __int64 *v77; // [rsp+18h] [rbp-198h]
  __int64 v79; // [rsp+28h] [rbp-188h]
  __int64 *v80; // [rsp+40h] [rbp-170h]
  unsigned int v82; // [rsp+50h] [rbp-160h]
  unsigned int v83; // [rsp+50h] [rbp-160h]
  unsigned int v84; // [rsp+50h] [rbp-160h]
  unsigned int v85; // [rsp+50h] [rbp-160h]
  __int64 v86; // [rsp+58h] [rbp-158h]
  const __m128i *v87; // [rsp+60h] [rbp-150h]
  __m128i *v88; // [rsp+68h] [rbp-148h]
  unsigned int v89; // [rsp+68h] [rbp-148h]
  int v90; // [rsp+68h] [rbp-148h]
  int v91; // [rsp+68h] [rbp-148h]
  int v92; // [rsp+68h] [rbp-148h]
  __m128i v93; // [rsp+70h] [rbp-140h] BYREF
  __m128i v94; // [rsp+80h] [rbp-130h] BYREF
  __m128i v95; // [rsp+90h] [rbp-120h] BYREF
  __m128i v96; // [rsp+A0h] [rbp-110h]
  __int64 v97; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-F8h]
  void *v99; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v100; // [rsp+C8h] [rbp-E8h]
  _OWORD v101[2]; // [rsp+D0h] [rbp-E0h] BYREF
  void *src; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v103; // [rsp+F8h] [rbp-B8h]
  _BYTE v104[176]; // [rsp+100h] [rbp-B0h] BYREF

  v3 = *(__int64 **)a3;
  src = v104;
  v103 = 0x400000000LL;
  v80 = v3;
  v77 = &v3[2 * *(unsigned int *)(a3 + 8)];
  if ( v3 != v77 )
  {
    while ( 1 )
    {
      v86 = *v80;
      v79 = v80[1];
      if ( *(_DWORD *)(*v80 + 40) <= 1u )
        break;
      v4 = *(_QWORD *)(*v80 + 32);
      if ( *(_BYTE *)v4 || *(_DWORD *)(v4 + 8) )
        break;
      LODWORD(v103) = 0;
      v17 = v80 + 2;
LABEL_31:
      v80 = v17;
      if ( v77 == v17 )
      {
        if ( src != v104 )
          _libc_free((unsigned __int64)src);
        return;
      }
    }
    v5 = sub_1E16510(v86);
    v6 = (__m128i *)src;
    v7 = v5;
    v8 = 32LL * (unsigned int)v103;
    v9 = (const __m128i *)((char *)src + v8);
    v10 = v8 >> 5;
    v11 = v8 >> 7;
    v87 = v9;
    if ( v11 )
    {
      v88 = (__m128i *)((char *)src + 128 * v11);
      while ( 1 )
      {
        v12 = v6->m128i_i64[0];
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_8;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_8;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        v35 = v97;
        v82 = v98;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v12 + 24), *(unsigned __int64 **)(v12 + 32));
        if ( v82 + v35 > (unsigned int)v98 && (int)v97 + (int)v98 > v82 )
          goto LABEL_8;
        v36 = v6 + 2;
        v37 = v6[2].m128i_i64[0];
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_53;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v37 + 24), *(unsigned __int64 **)(v37 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_53;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        v83 = v98;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v37 + 24), *(unsigned __int64 **)(v37 + 32));
        if ( (int)v98 + (int)v97 > v83 )
          goto LABEL_53;
        v36 = v6 + 4;
        v38 = v6[4].m128i_i64[0];
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_53;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v38 + 24), *(unsigned __int64 **)(v38 + 32));
        if ( !(_BYTE)v99 )
          goto LABEL_53;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        v84 = v98;
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v38 + 24), *(unsigned __int64 **)(v38 + 32));
        if ( (int)v98 + (int)v97 > v84
          || (v36 = v6 + 6,
              v39 = v6[6].m128i_i64[0],
              sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32)),
              !(_BYTE)v99)
          || (sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v39 + 24), *(unsigned __int64 **)(v39 + 32)), !(_BYTE)v99)
          || (sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32)),
              v85 = v98,
              sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v39 + 24), *(unsigned __int64 **)(v39 + 32)),
              (int)v98 + (int)v97 > v85) )
        {
LABEL_53:
          v6 = v36;
          goto LABEL_8;
        }
        v6 += 8;
        if ( v88 == v6 )
        {
          v10 = ((char *)v87 - (char *)v6) >> 5;
          break;
        }
      }
    }
    if ( v10 != 2 )
    {
      if ( v10 != 3 )
      {
        if ( v10 != 1 )
        {
LABEL_52:
          v6 = (__m128i *)v87;
          goto LABEL_15;
        }
LABEL_102:
        v74 = v6->m128i_i64[0];
        sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        if ( (_BYTE)v99 )
        {
          sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v74 + 24), *(unsigned __int64 **)(v74 + 32));
          if ( (_BYTE)v99 )
          {
            sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
            v75 = v98;
            v92 = v97;
            sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v74 + 24), *(unsigned __int64 **)(v74 + 32));
            if ( (int)v98 + (int)v97 <= v75 || v92 + v75 <= (unsigned int)v98 )
              goto LABEL_52;
          }
        }
LABEL_8:
        if ( v87 != v6 )
        {
          for ( i = v6 + 2; v87 != i; i += 2 )
          {
            v14 = i->m128i_i64[0];
            sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
            if ( (_BYTE)v99 )
            {
              sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v14 + 24), *(unsigned __int64 **)(v14 + 32));
              if ( (_BYTE)v99 )
              {
                sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
                v89 = v98;
                sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v14 + 24), *(unsigned __int64 **)(v14 + 32));
                if ( (int)v98 + (int)v97 <= v89 )
                {
                  v6 += 2;
                  v6[-2] = _mm_loadu_si128(i);
                  v6[-1] = _mm_loadu_si128(i + 1);
                }
              }
            }
          }
        }
LABEL_15:
        LODWORD(v103) = ((char *)v6 - (_BYTE *)src) >> 5;
        v15 = sub_397FAE0(a1, v86);
        if ( v79 )
        {
          v16 = sub_397FB50(a1, v79);
          v17 = v80 + 2;
        }
        else
        {
          v17 = v80 + 2;
          if ( (__int64 *)(*(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8)) == v80 + 2 )
            v16 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 392LL);
          else
            v16 = sub_397FAE0(a1, v80[2]);
        }
        v18 = sub_1E16510(v86);
        v19 = *(char **)(v86 + 32);
        v20 = *v19;
        if ( *v19 )
        {
          v40 = *((_QWORD *)v19 + 3);
          v93.m128i_i64[0] = v18;
          if ( v20 == 1 )
          {
            v93.m128i_i32[2] = 1;
            v94.m128i_i8[8] = 0;
            v94.m128i_i32[3] = 0;
            v94.m128i_i64[0] = v40;
          }
          else
          {
            if ( v20 == 3 )
              v93.m128i_i32[2] = 2;
            else
              v93.m128i_i32[2] = 3;
            v94.m128i_i8[8] = 0;
            v94.m128i_i32[3] = 0;
            v94.m128i_i64[0] = v40;
          }
        }
        else
        {
          v21 = *((_DWORD *)v19 + 2);
          v22 = v19[40] == 1;
          v93.m128i_i64[0] = v18;
          v93.m128i_i32[2] = 0;
          v94.m128i_i32[3] = v21;
          v94.m128i_i8[8] = !v22;
        }
        v23 = _mm_loadu_si128(&v93);
        v97 = v15;
        v24 = _mm_loadu_si128(&v94);
        v98 = v16;
        v99 = v101;
        v100 = 0x100000001LL;
        v95 = v23;
        v96 = v24;
        v101[0] = v23;
        v101[1] = v24;
        sub_15B1350((__int64)&v95, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
        v27 = (unsigned int)v103;
        if ( v96.m128i_i8[0] )
        {
          if ( (unsigned int)v103 >= HIDWORD(v103) )
          {
            sub_16CD150((__int64)&src, v104, 0, 32, v25, v26);
            v27 = (unsigned int)v103;
          }
          v41 = _mm_loadu_si128(&v94);
          v42 = (__m128i *)((char *)src + 32 * v27);
          *v42 = _mm_loadu_si128(&v93);
          v42[1] = v41;
          v43 = *(unsigned int *)(a2 + 8);
          v27 = (unsigned int)(v103 + 1);
          LODWORD(v103) = v103 + 1;
          if ( (_DWORD)v43 )
          {
            if ( (unsigned __int8)sub_3990E70(*(_QWORD *)a2 + (v43 << 6) - 64, (__int64)&v97) )
            {
              v31 = *(_QWORD *)a2;
              v33 = *(unsigned int *)(a2 + 8);
LABEL_27:
              v34 = v31 + (v33 << 6);
              if ( v31 != v34 - 64 && *(_QWORD *)(v34 - 120) == *(_QWORD *)(v34 - 64) )
              {
                v62 = *(unsigned int *)(v34 - 104);
                if ( v62 == *(_DWORD *)(v34 - 40) )
                {
                  v63 = *(_QWORD *)(v34 - 112);
                  v64 = *(_QWORD *)(v34 - 48);
                  v65 = v63 + 32 * v62;
                  if ( v63 == v65 )
                  {
LABEL_87:
                    *(_QWORD *)(v34 - 120) = *(_QWORD *)(v34 - 56);
                    v67 = (unsigned int)(*(_DWORD *)(a2 + 8) - 1);
                    *(_DWORD *)(a2 + 8) = v67;
                    v68 = *(_QWORD *)a2 + (v67 << 6);
                    v69 = *(_QWORD *)(v68 + 16);
                    if ( v69 != v68 + 32 )
                      _libc_free(v69);
                  }
                  else
                  {
                    while ( 1 )
                    {
                      v66 = *(_DWORD *)(v63 + 8);
                      if ( v66 != *(_DWORD *)(v64 + 8) || *(_QWORD *)v63 != *(_QWORD *)v64 )
                        break;
                      if ( v66 )
                      {
                        if ( *(_QWORD *)(v63 + 16) != *(_QWORD *)(v64 + 16) )
                          break;
                      }
                      else if ( *(_BYTE *)(v63 + 24) != *(_BYTE *)(v64 + 24)
                             || *(_DWORD *)(v63 + 28) != *(_DWORD *)(v64 + 28) )
                      {
                        break;
                      }
                      v63 += 32;
                      v64 += 32;
                      if ( v65 == v63 )
                        goto LABEL_87;
                    }
                  }
                }
              }
              if ( v99 != v101 )
                _libc_free((unsigned __int64)v99);
              goto LABEL_31;
            }
            v27 = (unsigned int)v103;
          }
        }
        if ( v27 )
        {
          v44 = (unsigned int)v100;
          v45 = v27;
          v46 = src;
          v47 = 32 * v27;
          if ( v27 > HIDWORD(v100) - (unsigned __int64)(unsigned int)v100 )
          {
            sub_16CD150((__int64)&v99, v101, v27 + (unsigned int)v100, 32, v25, v26);
            v44 = (unsigned int)v100;
          }
          memcpy((char *)v99 + 32 * v44, v46, v47);
          v49 = (__m128i *)v99;
          LODWORD(v100) = v100 + v45;
          v50 = 32LL * (unsigned int)v100;
          v51 = (__m128i *)((char *)v99 + v50);
          v52 = (__m128i *)((char *)v99 + v50);
          if ( v99 != (char *)v99 + v50 )
          {
            _BitScanReverse64(&v53, v50 >> 5);
            sub_39908E0((__m128i *)v99, (__m128i *)((char *)v99 + v50), 2LL * (int)(63 - (v53 ^ 0x3F)), v48);
            if ( (unsigned __int64)v50 <= 0x200 )
            {
              sub_3985DD0(v49, v51);
            }
            else
            {
              v54 = v49 + 32;
              sub_3985DD0(v49, v49 + 32);
              if ( v51 != &v49[32] )
              {
                do
                {
                  v55 = v54;
                  v54 += 2;
                  sub_39856E0(v55);
                }
                while ( v51 != v54 );
              }
            }
            v51 = (__m128i *)v99;
            v52 = (__m128i *)((char *)v99 + 32 * (unsigned int)v100);
          }
          v56 = 0;
          v57 = sub_3984B10(v51, v52);
          v58 = v99;
          v59 = v57;
          if ( (char *)v99 + 32 * (unsigned int)v100 != (char *)v52 )
          {
            v56 = (_BYTE *)v99 + 32 * (unsigned int)v100 - (_BYTE *)v52;
            v60 = (__m128i *)memmove(v57, v52, v56);
            v58 = v99;
            v59 = v60;
          }
          LODWORD(v100) = (&v59->m128i_i8[v56] - v58) >> 5;
          v28 = a2;
          v29 = *(_DWORD *)(a2 + 8);
          if ( v29 < *(_DWORD *)(a2 + 12) )
            goto LABEL_22;
        }
        else
        {
          v28 = a2;
          v29 = *(_DWORD *)(a2 + 8);
          if ( v29 < *(_DWORD *)(a2 + 12) )
          {
LABEL_22:
            v30 = a2;
            v31 = *(_QWORD *)a2;
            v32 = (__int64 *)(*(_QWORD *)a2 + ((unsigned __int64)v29 << 6));
            if ( v32 )
            {
              *v32 = v97;
              v32[1] = v98;
              v32[2] = (__int64)(v32 + 4);
              v32[3] = 0x100000000LL;
              if ( (_DWORD)v100 )
              {
                sub_39849D0((__int64)(v32 + 2), (char **)&v99, v31, a2, v25, v26);
                v30 = a2;
              }
              v29 = *(_DWORD *)(v30 + 8);
              v31 = *(_QWORD *)v30;
            }
            v33 = v29 + 1;
            *(_DWORD *)(a2 + 8) = v33;
            goto LABEL_27;
          }
        }
        v61 = v28;
        sub_398EB30(v28, 0);
        v29 = *(_DWORD *)(v61 + 8);
        goto LABEL_22;
      }
      v70 = v6->m128i_i64[0];
      sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
      if ( !(_BYTE)v99 )
        goto LABEL_8;
      sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v70 + 24), *(unsigned __int64 **)(v70 + 32));
      if ( !(_BYTE)v99 )
        goto LABEL_8;
      sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
      v71 = v98;
      v90 = v97;
      sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v70 + 24), *(unsigned __int64 **)(v70 + 32));
      if ( (int)v98 + (int)v97 > v71 && v90 + v71 > (unsigned int)v98 )
        goto LABEL_8;
      v6 += 2;
    }
    v72 = v6->m128i_i64[0];
    sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
    if ( !(_BYTE)v99 )
      goto LABEL_8;
    sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v72 + 24), *(unsigned __int64 **)(v72 + 32));
    if ( !(_BYTE)v99 )
      goto LABEL_8;
    sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v7 + 24), *(unsigned __int64 **)(v7 + 32));
    v73 = v98;
    v91 = v97;
    sub_15B1350((__int64)&v97, *(unsigned __int64 **)(v72 + 24), *(unsigned __int64 **)(v72 + 32));
    if ( (int)v98 + (int)v97 > v73 && v91 + v73 > (unsigned int)v98 )
      goto LABEL_8;
    v6 += 2;
    goto LABEL_102;
  }
}

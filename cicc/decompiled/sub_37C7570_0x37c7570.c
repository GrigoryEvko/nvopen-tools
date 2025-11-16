// Function: sub_37C7570
// Address: 0x37c7570
//
__int64 __fastcall sub_37C7570(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r13
  __int64 v10; // r14
  __int16 *v11; // r8
  unsigned int v12; // esi
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // r8
  const __m128i *v18; // rax
  __m128i *v19; // rdx
  unsigned int v20; // r13d
  __int64 v21; // r15
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // r8
  __int64 v33; // rax
  const __m128i *v34; // r13
  __m128i *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rax
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rcx
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  const __m128i *v47; // rdx
  __m128i *v48; // rax
  _QWORD *v49; // rdi
  char *v50; // rax
  __int64 v51; // rdx
  unsigned __int16 *v52; // r14
  char *i; // r12
  __int64 v54; // rdi
  __int64 v55; // rsi
  _DWORD *v56; // r13
  __int64 v57; // rax
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // r8
  int v60; // r15d
  char v61; // al
  __int64 v62; // rcx
  __int16 *v63; // r8
  __int16 *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r15
  __int64 v67; // rax
  __int64 v68; // rdx
  __int64 v69; // rcx
  unsigned __int64 v70; // rax
  __int16 *v71; // rax
  unsigned __int16 v72; // dx
  const void *v73; // rsi
  char *v74; // r12
  unsigned __int64 v75; // r15
  const void *v76; // rsi
  unsigned __int64 v77; // r12
  const void *v78; // rsi
  unsigned __int64 v79; // r12
  const void *v80; // rsi
  __int128 v81; // [rsp-D8h] [rbp-D8h]
  __int64 v82; // [rsp-B0h] [rbp-B0h]
  __int16 *v83; // [rsp-B0h] [rbp-B0h]
  __int16 *v84; // [rsp-A8h] [rbp-A8h]
  unsigned int *v85; // [rsp-A8h] [rbp-A8h]
  __int64 (__fastcall *v86)(__int64, __int64, _QWORD, unsigned int *); // [rsp-A0h] [rbp-A0h]
  unsigned int v87; // [rsp-98h] [rbp-98h] BYREF
  int v88; // [rsp-94h] [rbp-94h] BYREF
  __int64 v89; // [rsp-90h] [rbp-90h]
  __int64 v90; // [rsp-88h] [rbp-88h]
  __int64 v91; // [rsp-80h] [rbp-80h]
  __int64 v92; // [rsp-78h] [rbp-78h]
  __int16 *v93; // [rsp-68h] [rbp-68h] BYREF
  __int64 v94; // [rsp-60h] [rbp-60h]
  __int128 v95; // [rsp-58h] [rbp-58h]
  __int64 v96; // [rsp-48h] [rbp-48h]

  result = 0;
  if ( *(_WORD *)(a2 + 68) == 17 )
  {
    result = 1;
    if ( !*(_QWORD *)(a1 + 424) && !*(_QWORD *)(a1 + 432) )
    {
      v8 = *(_QWORD *)(a2 + 32);
      v10 = a1 + 776;
      v11 = (__int16 *)*(unsigned int *)(v8 + 64);
      if ( !*(_BYTE *)v8 )
      {
        v12 = *(_DWORD *)(v8 + 8);
        if ( v12 )
        {
          v37 = *(_QWORD *)(a1 + 408);
          v36 = v37;
          v38 = *(_DWORD *)(*(_QWORD *)(v37 + 64) + 4LL * v12);
          if ( v38 == -1 )
          {
            v83 = (__int16 *)*(unsigned int *)(v8 + 64);
            v85 = (unsigned int *)(*(_QWORD *)(v37 + 64) + 4LL * v12);
            v38 = sub_37BA230(v37, v12);
            v11 = v83;
            *v85 = v38;
            v37 = *(_QWORD *)(a1 + 408);
          }
          v39 = *(_QWORD *)(*(_QWORD *)(v36 + 32) + 8LL * v38);
          v40 = *(_QWORD *)(a2 + 24);
          v93 = v11;
          BYTE8(v95) = 1;
          v94 = v40;
          *(_QWORD *)&v95 = v39;
          v41 = sub_37BA440(v37, v12);
          v44 = *(unsigned int *)(a1 + 784);
          BYTE4(v96) = 1;
          LODWORD(v96) = v41;
          v45 = v44 + 1;
          if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 788) )
          {
            v75 = *(_QWORD *)(a1 + 776);
            v76 = (const void *)(a1 + 792);
            if ( v75 <= (unsigned __int64)&v93 && (unsigned __int64)&v93 < v75 + 40 * v44 )
            {
              sub_C8D5F0(v10, v76, v45, 0x28u, v42, v43);
              v46 = *(_QWORD *)(a1 + 776);
              v44 = *(unsigned int *)(a1 + 784);
              v47 = (const __m128i *)((char *)&v93 + v46 - v75);
            }
            else
            {
              sub_C8D5F0(v10, v76, v45, 0x28u, v42, v43);
              v44 = *(unsigned int *)(a1 + 784);
              v46 = *(_QWORD *)(a1 + 776);
              v47 = (const __m128i *)&v93;
            }
          }
          else
          {
            v46 = *(_QWORD *)(a1 + 776);
            v47 = (const __m128i *)&v93;
          }
          v48 = (__m128i *)(v46 + 40 * v44);
          *v48 = _mm_loadu_si128(v47);
          v48[1] = _mm_loadu_si128(v47 + 1);
          v48[2].m128i_i64[0] = v47[2].m128i_i64[0];
          v49 = *(_QWORD **)(a1 + 16);
          ++*(_DWORD *)(a1 + 784);
          v50 = sub_E922F0(v49, *(_DWORD *)(v8 + 8));
          v52 = (unsigned __int16 *)&v50[2 * v51];
          for ( i = v50; v52 != (unsigned __int16 *)i; i += 2 )
          {
            v54 = *(_QWORD *)(a1 + 408);
            v55 = *(unsigned __int16 *)i;
            v56 = (_DWORD *)(*(_QWORD *)(v54 + 64) + 4 * v55);
            if ( *v56 == -1 )
              *v56 = sub_37BA230(v54, v55);
          }
          return 1;
        }
        goto LABEL_8;
      }
      if ( *(_BYTE *)v8 != 5 )
      {
LABEL_8:
        v13 = *(unsigned int *)(a1 + 784);
        v14 = *(_QWORD *)(a2 + 24);
        v93 = (__int16 *)*(unsigned int *)(v8 + 64);
        v15 = *(unsigned int *)(a1 + 788);
        v16 = *(_QWORD *)(a1 + 776);
        v96 = 0;
        v17 = v13 + 1;
        v94 = v14;
        v18 = (const __m128i *)&v93;
        v95 = 0;
        if ( v13 + 1 > v15 )
        {
          v73 = (const void *)(a1 + 792);
          if ( v16 > (unsigned __int64)&v93 || (unsigned __int64)&v93 >= v16 + 40 * v13 )
          {
            sub_C8D5F0(a1 + 776, v73, v17, 0x28u, v17, a6);
            v16 = *(_QWORD *)(a1 + 776);
            v13 = *(unsigned int *)(a1 + 784);
            v18 = (const __m128i *)&v93;
          }
          else
          {
            v74 = (char *)&v93 - v16;
            sub_C8D5F0(a1 + 776, v73, v17, 0x28u, v17, a6);
            v16 = *(_QWORD *)(a1 + 776);
            v13 = *(unsigned int *)(a1 + 784);
            v18 = (const __m128i *)&v74[v16];
          }
        }
        v19 = (__m128i *)(v16 + 40 * v13);
        *v19 = _mm_loadu_si128(v18);
        v19[1] = _mm_loadu_si128(v18 + 1);
        v19[2].m128i_i64[0] = v18[2].m128i_i64[0];
        ++*(_DWORD *)(a1 + 784);
        return 1;
      }
      v20 = *(_DWORD *)(v8 + 24);
      if ( *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL)
                     + 40LL * (*(_DWORD *)(*(_QWORD *)(a1 + 48) + 32LL) + v20)
                     + 8) == -1 )
      {
        v57 = *(_QWORD *)(a2 + 24);
        v30 = *(unsigned int *)(a1 + 784);
        v93 = v11;
        v96 = 0;
        v94 = v57;
        v58 = *(unsigned int *)(a1 + 788);
        v59 = v30 + 1;
        v95 = 0;
        if ( v30 + 1 <= v58 )
        {
          v33 = *(_QWORD *)(a1 + 776);
          v34 = (const __m128i *)&v93;
LABEL_15:
          v35 = (__m128i *)(v33 + 40 * v30);
          *v35 = _mm_loadu_si128(v34);
          v35[1] = _mm_loadu_si128(v34 + 1);
          v35[2].m128i_i64[0] = v34[2].m128i_i64[0];
          ++*(_DWORD *)(a1 + 784);
          return 1;
        }
        v79 = *(_QWORD *)(a1 + 776);
        v34 = (const __m128i *)&v93;
        v80 = (const void *)(a1 + 792);
        if ( v79 > (unsigned __int64)&v93 )
        {
LABEL_47:
          sub_C8D5F0(v10, v80, v59, 0x28u, v59, a6);
          v30 = *(unsigned int *)(a1 + 784);
          v33 = *(_QWORD *)(a1 + 776);
          goto LABEL_15;
        }
      }
      else
      {
        v21 = *(_QWORD *)(a1 + 40);
        v87 = 0;
        v84 = v11;
        v86 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD, unsigned int *))(*(_QWORD *)v21 + 224LL);
        v22 = sub_2E88D60(a2);
        v23 = v86(v21, v22, v20, &v87);
        v24 = *(_QWORD *)(a1 + 408);
        LODWORD(v90) = v87;
        *((_QWORD *)&v81 + 1) = v23;
        *(_QWORD *)&v81 = v90;
        v92 = v25;
        v91 = v23;
        v89 = sub_37C6BA0(v24, v22, v25, v87, v26, v27, v81, v25);
        if ( !BYTE4(v89) )
        {
          v29 = *(_QWORD *)(a2 + 24);
          v30 = *(unsigned int *)(a1 + 784);
          v93 = v84;
          v96 = 0;
          v94 = v29;
          v31 = *(unsigned int *)(a1 + 788);
          v32 = v30 + 1;
          v95 = 0;
          if ( v30 + 1 > v31 )
          {
            v77 = *(_QWORD *)(a1 + 776);
            v34 = (const __m128i *)&v93;
            v78 = (const void *)(a1 + 792);
            if ( v77 > (unsigned __int64)&v93 || (unsigned __int64)&v93 >= v77 + 40 * v30 )
            {
              sub_C8D5F0(v10, v78, v30 + 1, 0x28u, v32, v28);
              v30 = *(unsigned int *)(a1 + 784);
              v33 = *(_QWORD *)(a1 + 776);
            }
            else
            {
              sub_C8D5F0(v10, v78, v30 + 1, 0x28u, v32, v28);
              v33 = *(_QWORD *)(a1 + 776);
              v30 = *(unsigned int *)(a1 + 784);
              v34 = (const __m128i *)((char *)&v93 + v33 - v77);
            }
          }
          else
          {
            v33 = *(_QWORD *)(a1 + 776);
            v34 = (const __m128i *)&v93;
          }
          goto LABEL_15;
        }
        v34 = (const __m128i *)&v93;
        v82 = *(_QWORD *)(a1 + 408);
        v88 = (unsigned __int16)*(_QWORD *)(*(_QWORD *)(a2 + 32) + 104LL);
        v60 = *(_DWORD *)(v82 + 288) * (v89 - 1);
        v61 = sub_37BD660(v82 + 824, (unsigned __int16 *)&v88, &v93);
        v62 = v82;
        v63 = v84;
        if ( v61 )
        {
          v64 = v93 + 2;
        }
        else
        {
          v71 = sub_37C5CE0(v82 + 824, (unsigned __int16 *)&v88, v93);
          v72 = v88;
          v63 = v84;
          *((_DWORD *)v71 + 1) = 0;
          v62 = v82;
          v64 = v71 + 2;
          *((_DWORD *)v64 - 1) = __PAIR32__(HIWORD(v88), v72);
        }
        v65 = *(_QWORD *)(a1 + 408);
        v66 = (unsigned int)(*(_DWORD *)v64 + *(_DWORD *)(v62 + 284) + v60);
        v67 = *(_QWORD *)(v65 + 64);
        v68 = *(_QWORD *)(*(_QWORD *)(v65 + 32) + 8LL * *(unsigned int *)(v67 + 4 * v66));
        LODWORD(v67) = *(_DWORD *)(v67 + 4 * v66);
        v69 = *(_QWORD *)(a2 + 24);
        v93 = v63;
        LODWORD(v96) = v67;
        v70 = *(unsigned int *)(a1 + 788);
        *(_QWORD *)&v95 = v68;
        v30 = *(unsigned int *)(a1 + 784);
        v94 = v69;
        v59 = v30 + 1;
        BYTE8(v95) = 1;
        BYTE4(v96) = 1;
        if ( v30 + 1 <= v70 )
        {
          v33 = *(_QWORD *)(a1 + 776);
          goto LABEL_15;
        }
        v79 = *(_QWORD *)(a1 + 776);
        v80 = (const void *)(a1 + 792);
        if ( v79 > (unsigned __int64)&v93 )
          goto LABEL_47;
      }
      if ( (unsigned __int64)&v93 < v79 + 40 * v30 )
      {
        sub_C8D5F0(v10, v80, v59, 0x28u, v59, a6);
        v33 = *(_QWORD *)(a1 + 776);
        v30 = *(unsigned int *)(a1 + 784);
        v34 = (const __m128i *)((char *)&v93 + v33 - v79);
        goto LABEL_15;
      }
      goto LABEL_47;
    }
  }
  return result;
}

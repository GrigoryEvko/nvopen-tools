// Function: sub_27A6600
// Address: 0x27a6600
//
void __fastcall sub_27A6600(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 *v4; // rax
  __int64 *v5; // rcx
  __int64 v6; // r14
  __int64 *v7; // r13
  __m128i *v9; // r12
  __int64 v10; // rdx
  int *m128i_i32; // rbx
  unsigned __int64 v12; // rax
  int v13; // edx
  unsigned __int64 v14; // rax
  bool v15; // cf
  __int64 v16; // rdx
  const __m128i *v17; // rsi
  __int64 v18; // rax
  const __m128i *v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int32 v22; // edx
  __int64 v23; // rdi
  const __m128i *v24; // rbx
  const __m128i *v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rsi
  __int32 v29; // esi
  __int64 v30; // rdi
  __int64 v31; // rcx
  const __m128i *v32; // rax
  char v33; // al
  __int64 v34; // r9
  char **v35; // r12
  char **v36; // r12
  __int64 v37; // rdx
  unsigned __int64 v38; // r8
  unsigned __int64 v39; // rdi
  unsigned __int64 v40; // rsi
  int v41; // eax
  char **v42; // rdi
  _BYTE *v43; // rdi
  __int64 v44; // rax
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 v47; // rdx
  const void *v48; // rsi
  __int64 v49; // rbx
  __int64 v50; // r12
  char **v51; // r14
  __int64 v52; // r15
  __int64 v53; // r13
  __int64 v54; // r8
  __int64 v55; // rdi
  const __m128i *v56; // r8
  char *v57; // r12
  __int32 v58; // eax
  __int32 v59; // eax
  __int64 v60; // [rsp+10h] [rbp-140h]
  __int64 v61; // [rsp+18h] [rbp-138h]
  const __m128i *v62; // [rsp+20h] [rbp-130h]
  __int64 *v63; // [rsp+28h] [rbp-128h]
  __int64 v66; // [rsp+40h] [rbp-110h]
  __int64 *v67; // [rsp+40h] [rbp-110h]
  __int64 v68; // [rsp+48h] [rbp-108h]
  __int64 v69; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v70; // [rsp+98h] [rbp-B8h]
  __int64 v71; // [rsp+A0h] [rbp-B0h]
  _BYTE v72[40]; // [rsp+A8h] [rbp-A8h] BYREF
  char **v73; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v74; // [rsp+D8h] [rbp-78h]
  char *v75[14]; // [rsp+E0h] [rbp-70h] BYREF

  if ( *(_DWORD *)(a2 + 16) )
  {
    v4 = *(__int64 **)(a2 + 8);
    v5 = &v4[11 * *(unsigned int *)(a2 + 24)];
    v63 = v5;
    if ( v4 != v5 )
    {
      while ( *v4 == -4096 || *v4 == -8192 )
      {
        v4 += 11;
        if ( v5 == v4 )
          return;
      }
      if ( v5 != v4 )
      {
        v6 = *v4;
        v7 = v4;
        while ( 1 )
        {
          v9 = (__m128i *)v7[1];
          v10 = 2LL * *((unsigned int *)v7 + 4);
          m128i_i32 = v9[v10].m128i_i32;
          sub_27A3970((__int64 *)&v73, v9, (v10 * 16) >> 5);
          if ( v75[0] )
            sub_27A2070(v9->m128i_i32, m128i_i32, v75[0], v74);
          else
            sub_27A1900((__int64)v9, (__int64)m128i_i32);
          j_j___libc_free_0((unsigned __int64)v75[0]);
          v12 = *(_QWORD *)(v6 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v12 == v6 + 48 )
          {
            v68 = 0;
          }
          else
          {
            if ( !v12 )
              BUG();
            v13 = *(unsigned __int8 *)(v12 - 24);
            v14 = v12 - 24;
            v15 = (unsigned int)(v13 - 30) < 0xB;
            v16 = 0;
            if ( v15 )
              v16 = v14;
            v68 = v16;
          }
          v17 = (const __m128i *)v7[1];
          v18 = 32LL * *((unsigned int *)v7 + 4);
          v19 = &v17[(unsigned __int64)v18 / 0x10];
          v20 = v18 >> 5;
          v21 = v18 >> 7;
          if ( !v21 )
            break;
          v22 = v17->m128i_i32[0];
          v23 = v17->m128i_i64[1];
          v24 = (const __m128i *)v7[1];
          v25 = &v17[8 * v21];
          while ( v24->m128i_i64[1] == v23 )
          {
            v56 = v24 + 2;
            if ( v22 != v24[2].m128i_i32[0]
              || v23 != v24[2].m128i_i64[1]
              || (v56 = v24 + 4, v22 != v24[4].m128i_i32[0])
              || v23 != v24[4].m128i_i64[1]
              || (v56 = v24 + 6, v22 != v24[6].m128i_i32[0])
              || v23 != v24[6].m128i_i64[1] )
            {
              v24 = v56;
              break;
            }
            v24 += 8;
            if ( v25 == v24 )
            {
              v20 = ((char *)v19 - (char *)v24) >> 5;
              goto LABEL_96;
            }
            if ( v22 != v24->m128i_i32[0] )
              break;
          }
LABEL_20:
          if ( v17 != v24 )
          {
            while ( 1 )
            {
              v73 = v75;
              v74 = 0x200000000LL;
              sub_27A6190(a1, v17, v24, v6, a3, (__int64)&v73);
              v33 = sub_27A27B0(a1, (__int64)v73, (__int64)&v73[4 * (unsigned int)v74], v68);
              v35 = v73;
              if ( !v33 )
                goto LABEL_22;
              v69 = v6;
              v36 = (char **)&v69;
              v37 = *(unsigned int *)(a4 + 8);
              v38 = *(unsigned int *)(a4 + 12);
              v71 = 0x400000000LL;
              v39 = *(_QWORD *)a4;
              v40 = v37 + 1;
              v70 = v72;
              v41 = v37;
              if ( v37 + 1 > v38 )
              {
                if ( v39 > (unsigned __int64)&v69 )
                {
                  v55 = a4;
                }
                else
                {
                  if ( (unsigned __int64)&v69 < v39 + 56 * v37 )
                  {
                    v57 = (char *)&v69 - v39;
                    sub_27A3A70(a4, v40, v37, (__int64)v72, v38, v34);
                    v39 = *(_QWORD *)a4;
                    v37 = *(unsigned int *)(a4 + 8);
                    v36 = (char **)&v57[*(_QWORD *)a4];
                    v41 = *(_DWORD *)(a4 + 8);
                    goto LABEL_31;
                  }
                  v55 = a4;
                }
                sub_27A3A70(v55, v40, v37, (__int64)v72, v38, v34);
                v37 = *(unsigned int *)(a4 + 8);
                v39 = *(_QWORD *)a4;
                v41 = *(_DWORD *)(a4 + 8);
              }
LABEL_31:
              v42 = (char **)(v39 + 56 * v37);
              if ( v42 )
              {
                *v42 = *v36;
                v42[1] = (char *)(v42 + 3);
                v42[2] = (char *)0x400000000LL;
                if ( *((_DWORD *)v36 + 4) )
                  sub_27A0EB0((__int64)(v42 + 1), v36 + 1, v37, (__int64)v72, v38, v34);
                v41 = *(_DWORD *)(a4 + 8);
              }
              v43 = v70;
              v44 = (unsigned int)(v41 + 1);
              *(_DWORD *)(a4 + 8) = v44;
              if ( v43 != v72 )
              {
                _libc_free((unsigned __int64)v43);
                v44 = *(unsigned int *)(a4 + 8);
              }
              v35 = v73;
              v45 = (__int64)&v73[4 * (unsigned int)v74];
              v46 = *(_QWORD *)a4 + 56 * v44 - 56;
              if ( (char **)v45 != v73 )
              {
                v67 = v7;
                v47 = *(unsigned int *)(v46 + 16);
                v48 = (const void *)(v46 + 24);
                v62 = v24;
                v49 = (__int64)v73;
                v50 = v46;
                v61 = v6;
                v51 = &v73[4 * (unsigned int)v74];
                v60 = a1;
                v52 = v46 + 8;
                do
                {
                  v53 = *(_QWORD *)(v49 + 24);
                  if ( v47 + 1 > (unsigned __int64)*(unsigned int *)(v50 + 20) )
                  {
                    sub_C8D5F0(v52, v48, v47 + 1, 8u, v38, v45);
                    v47 = *(unsigned int *)(v50 + 16);
                  }
                  v49 += 32;
                  *(_QWORD *)(*(_QWORD *)(v50 + 8) + 8 * v47) = v53;
                  v47 = (unsigned int)(*(_DWORD *)(v50 + 16) + 1);
                  *(_DWORD *)(v50 + 16) = v47;
                }
                while ( (char **)v49 != v51 );
                v7 = v67;
                v24 = v62;
                v6 = v61;
                a1 = v60;
                v35 = v73;
              }
LABEL_22:
              v26 = v7[1] + 32LL * *((unsigned int *)v7 + 4) - (_QWORD)v24;
              v27 = v26 >> 7;
              v28 = v26 >> 5;
              if ( v27 <= 0 )
              {
                v31 = (__int64)v24;
LABEL_62:
                switch ( v28 )
                {
                  case 2LL:
                    v58 = v24->m128i_i32[0];
                    break;
                  case 3LL:
                    v58 = v24->m128i_i32[0];
                    if ( *(_DWORD *)v31 != v24->m128i_i32[0] || *(_QWORD *)(v31 + 8) != v24->m128i_i64[1] )
                      goto LABEL_25;
                    v31 += 32;
                    break;
                  case 1LL:
                    v58 = v24->m128i_i32[0];
LABEL_83:
                    if ( *(_DWORD *)v31 == v58 && *(_QWORD *)(v31 + 8) == v24->m128i_i64[1] )
                      v31 = v7[1] + 32LL * *((unsigned int *)v7 + 4);
                    goto LABEL_25;
                  default:
                    v31 = v7[1] + 32LL * *((unsigned int *)v7 + 4);
                    goto LABEL_25;
                }
                if ( *(_DWORD *)v31 != v58 || *(_QWORD *)(v31 + 8) != v24->m128i_i64[1] )
                  goto LABEL_25;
                v31 += 32;
                goto LABEL_83;
              }
              v29 = v24->m128i_i32[0];
              v30 = v24->m128i_i64[1];
              v31 = (__int64)v24;
              v32 = &v24[8 * v27];
              while ( *(_QWORD *)(v31 + 8) == v30 )
              {
                v54 = v31 + 32;
                if ( v29 != *(_DWORD *)(v31 + 32)
                  || v30 != *(_QWORD *)(v31 + 40)
                  || (v54 = v31 + 64, v29 != *(_DWORD *)(v31 + 64))
                  || v30 != *(_QWORD *)(v31 + 72)
                  || (v54 = v31 + 96, v29 != *(_DWORD *)(v31 + 96))
                  || v30 != *(_QWORD *)(v31 + 104) )
                {
                  v31 = v54;
                  break;
                }
                v31 += 128;
                if ( v32 == (const __m128i *)v31 )
                {
                  v28 = (v7[1] + 32LL * *((unsigned int *)v7 + 4) - v31) >> 5;
                  goto LABEL_62;
                }
                if ( v29 != *(_DWORD *)v31 )
                  break;
              }
LABEL_25:
              if ( v35 != v75 )
              {
                v66 = v31;
                _libc_free((unsigned __int64)v35);
                v31 = v66;
              }
              v17 = v24;
              if ( (const __m128i *)v31 == v24 )
                break;
              v24 = (const __m128i *)v31;
            }
          }
          v7 += 11;
          if ( v7 != v63 )
          {
            while ( 1 )
            {
              v6 = *v7;
              if ( *v7 != -8192 && v6 != -4096 )
                break;
              v7 += 11;
              if ( v63 == v7 )
                return;
            }
            if ( v63 != v7 )
              continue;
          }
          return;
        }
        v24 = (const __m128i *)v7[1];
LABEL_96:
        switch ( v20 )
        {
          case 2LL:
            v59 = v17->m128i_i32[0];
            break;
          case 3LL:
            v59 = v17->m128i_i32[0];
            if ( v24->m128i_i32[0] != v17->m128i_i32[0] || v24->m128i_i64[1] != v17->m128i_i64[1] )
              goto LABEL_20;
            v24 += 2;
            break;
          case 1LL:
            v59 = v17->m128i_i32[0];
LABEL_101:
            if ( v24->m128i_i32[0] == v59 && v24->m128i_i64[1] == v17->m128i_i64[1] )
              v24 = v19;
            goto LABEL_20;
          default:
            v24 = v19;
            goto LABEL_20;
        }
        if ( v24->m128i_i32[0] != v59 || v24->m128i_i64[1] != v17->m128i_i64[1] )
          goto LABEL_20;
        v24 += 2;
        goto LABEL_101;
      }
    }
  }
}

// Function: sub_141E480
// Address: 0x141e480
//
void __fastcall sub_141E480(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rdx
  __m128i v9; // xmm6
  __m128i v10; // xmm7
  __m128i v11; // xmm0
  __m128i v12; // xmm1
  unsigned __int64 v13; // rax
  __int64 v14; // r14
  bool v15; // zf
  bool v16; // al
  __int64 v17; // rax
  __int64 v18; // rsi
  int v19; // r10d
  unsigned int v20; // ecx
  __int64 *v21; // r15
  __int64 v22; // r8
  __int64 v23; // rax
  __m128i *v24; // rax
  unsigned int v25; // esi
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  unsigned int v28; // edx
  __int64 v29; // rbx
  __int64 v30; // r8
  _QWORD *v31; // rax
  __int64 v32; // rax
  char v33; // al
  unsigned int v34; // eax
  __int64 v35; // rax
  __m128i *v36; // rax
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rcx
  int v40; // edx
  __int64 v41; // rax
  __m128i v42; // xmm5
  __m128i *v43; // rax
  __m128i v44; // xmm4
  __m128i v45; // xmm5
  _QWORD *v46; // rdx
  _QWORD *v47; // rcx
  __int64 v48; // rdx
  int v49; // ecx
  int v50; // r10d
  __int64 v51; // r9
  int v52; // edi
  char v53; // [rsp+Bh] [rbp-F5h]
  char v54; // [rsp+Bh] [rbp-F5h]
  unsigned __int8 v55; // [rsp+Ch] [rbp-F4h]
  __m128i v56; // [rsp+10h] [rbp-F0h] BYREF
  __int64 v57; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v58; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v59; // [rsp+38h] [rbp-C8h]
  __int64 v60; // [rsp+40h] [rbp-C0h]
  int v61; // [rsp+48h] [rbp-B8h]
  __m128i v62; // [rsp+50h] [rbp-B0h] BYREF
  __m128i v63; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v64; // [rsp+70h] [rbp-90h]
  __m128i v65; // [rsp+80h] [rbp-80h] BYREF
  __m128i v66; // [rsp+90h] [rbp-70h] BYREF
  _QWORD *v67; // [rsp+A0h] [rbp-60h]
  __int64 v68; // [rsp+A8h] [rbp-58h]
  _QWORD v69[10]; // [rsp+B0h] [rbp-50h] BYREF

  v55 = a4;
  v8 = *(unsigned __int8 *)(a2 + 16);
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case '6':
      sub_141EB40(&v62, a2, v8, a4, a5);
      goto LABEL_42;
    case '7':
      sub_141EDF0(&v62, a2, v8, a4, a5);
      goto LABEL_3;
    case ':':
      sub_141F110(&v62);
LABEL_3:
      v9 = _mm_loadu_si128(&v62);
      v10 = _mm_loadu_si128(&v63);
      LOBYTE(v8) = *(_BYTE *)(a2 + 16);
      v67 = (_QWORD *)v64;
      v65 = v9;
      v66 = v10;
      break;
    case ';':
      sub_141F3C0(&v62);
      v44 = _mm_loadu_si128(&v63);
      LOBYTE(v8) = *(_BYTE *)(a2 + 16);
      v65 = _mm_loadu_si128(&v62);
      v67 = (_QWORD *)v64;
      v66 = v44;
      break;
    case 'R':
      sub_141F0A0(&v62);
LABEL_42:
      v45 = _mm_loadu_si128(&v63);
      LOBYTE(v8) = *(_BYTE *)(a2 + 16);
      v65 = _mm_loadu_si128(&v62);
      v67 = (_QWORD *)v64;
      v66 = v45;
      break;
    default:
      break;
  }
  v11 = _mm_loadu_si128(&v65);
  v12 = _mm_loadu_si128(&v66);
  v13 = (unsigned __int64)v67;
  v14 = *(_QWORD *)(a2 + 40);
  *(_DWORD *)(a3 + 8) = 0;
  v15 = *(_BYTE *)(a2 + 16) == 54;
  v62 = v11;
  v64 = v13;
  v63 = v12;
  if ( !v15 || (v53 = v8, v16 = sub_14152B0(a1 + 832, a2), LOBYTE(v8) = v53, !v16) )
  {
    v17 = *(unsigned int *)(a1 + 56);
    if ( (_DWORD)v17 )
    {
      v18 = *(_QWORD *)(a1 + 40);
      v19 = 1;
      v20 = (v17 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = (__int64 *)(v18 + 32LL * v20);
      v22 = *v21;
      if ( a2 == *v21 )
      {
LABEL_8:
        if ( v21 != (__int64 *)(v18 + 32 * v17) )
        {
          v23 = *(unsigned int *)(a3 + 8);
          if ( (unsigned int)v23 >= *(_DWORD *)(a3 + 12) )
          {
            sub_16CD150(a3, a3 + 16, 0, 24);
            v23 = *(unsigned int *)(a3 + 8);
          }
          v24 = (__m128i *)(*(_QWORD *)a3 + 24 * v23);
          *v24 = _mm_loadu_si128((const __m128i *)(v21 + 1));
          v24[1].m128i_i64[0] = v21[3];
          ++*(_DWORD *)(a3 + 8);
          v25 = *(_DWORD *)(a1 + 88);
          v26 = v21[2] & 0xFFFFFFFFFFFFFFF8LL;
          if ( ((unsigned __int8)v21[2] & 7u) >= 3 )
            v26 = 0;
          v58 = v26;
          if ( v25 )
          {
            v27 = *(_QWORD *)(a1 + 72);
            v28 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v29 = v27 + 80LL * v28;
            v30 = *(_QWORD *)v29;
            if ( v26 == *(_QWORD *)v29 )
            {
LABEL_15:
              v31 = *(_QWORD **)(v29 + 16);
              if ( *(_QWORD **)(v29 + 24) != v31 )
              {
                v31 = (_QWORD *)sub_16CC9F0(v29 + 8, a2);
                if ( a2 == *v31 )
                {
                  v48 = *(_QWORD *)(v29 + 24);
                  if ( v48 == *(_QWORD *)(v29 + 16) )
                    v47 = (_QWORD *)(v48 + 8LL * *(unsigned int *)(v29 + 36));
                  else
                    v47 = (_QWORD *)(v48 + 8LL * *(unsigned int *)(v29 + 32));
                }
                else
                {
                  v32 = *(_QWORD *)(v29 + 24);
                  if ( v32 != *(_QWORD *)(v29 + 16) )
                  {
LABEL_18:
                    *v21 = -16;
                    --*(_DWORD *)(a1 + 48);
                    ++*(_DWORD *)(a1 + 52);
                    return;
                  }
                  v31 = (_QWORD *)(v32 + 8LL * *(unsigned int *)(v29 + 36));
                  v47 = v31;
                }
LABEL_49:
                if ( v47 != v31 )
                {
                  *v31 = -2;
                  ++*(_DWORD *)(v29 + 40);
                }
                goto LABEL_18;
              }
              v46 = &v31[*(unsigned int *)(v29 + 36)];
              v47 = v46;
              if ( v46 != v31 )
              {
                while ( a2 != *v31 )
                {
                  if ( v46 == ++v31 )
                    goto LABEL_59;
                }
                goto LABEL_49;
              }
LABEL_59:
              v31 = v46;
              goto LABEL_49;
            }
            v50 = 1;
            v51 = 0;
            while ( v30 != -8 )
            {
              if ( v30 == -16 && !v51 )
                v51 = v29;
              v28 = (v25 - 1) & (v50 + v28);
              v29 = v27 + 80LL * v28;
              v30 = *(_QWORD *)v29;
              if ( v26 == *(_QWORD *)v29 )
                goto LABEL_15;
              ++v50;
            }
            v52 = *(_DWORD *)(a1 + 80);
            if ( v51 )
              v29 = v51;
            ++*(_QWORD *)(a1 + 64);
            v49 = v52 + 1;
            if ( 4 * (v52 + 1) < 3 * v25 )
            {
              if ( v25 - *(_DWORD *)(a1 + 84) - v49 > v25 >> 3 )
                goto LABEL_56;
              goto LABEL_55;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 64);
          }
          v25 *= 2;
LABEL_55:
          sub_14171D0(a1 + 64, v25);
          sub_14153C0(a1 + 64, (__int64 *)&v58, &v65);
          v29 = v65.m128i_i64[0];
          v26 = v58;
          v49 = *(_DWORD *)(a1 + 80) + 1;
LABEL_56:
          *(_DWORD *)(a1 + 80) = v49;
          if ( *(_QWORD *)v29 != -8 )
            --*(_DWORD *)(a1 + 84);
          v47 = (_QWORD *)(v29 + 48);
          *(_QWORD *)v29 = v26;
          *(_QWORD *)(v29 + 8) = 0;
          v46 = (_QWORD *)(v29 + 48);
          *(_QWORD *)(v29 + 16) = v29 + 48;
          *(_QWORD *)(v29 + 24) = v29 + 48;
          *(_QWORD *)(v29 + 32) = 4;
          *(_DWORD *)(v29 + 40) = 0;
          goto LABEL_59;
        }
      }
      else
      {
        while ( v22 != -8 )
        {
          v20 = (v17 - 1) & (v19 + v20);
          v21 = (__int64 *)(v18 + 32LL * v20);
          v22 = *v21;
          if ( a2 == *v21 )
            goto LABEL_8;
          ++v19;
        }
      }
    }
    v33 = *(_BYTE *)(a2 + 16);
    if ( v33 == 54 || v33 == 55 )
    {
      v34 = *(unsigned __int16 *)(a2 + 18);
      if ( (v34 & 1) == 0 && ((v34 >> 7) & 6) == 0 )
      {
LABEL_32:
        v54 = v8;
        v37 = sub_157EB90(v14);
        v38 = sub_1632FA0(v37);
        v39 = *(_QWORD *)(a1 + 264);
        v67 = v69;
        v66.m128i_i64[0] = 0;
        v65.m128i_i64[1] = v38;
        v65.m128i_i64[0] = v62.m128i_i64[0];
        v66.m128i_i64[1] = v39;
        v68 = 0x400000000LL;
        if ( *(_BYTE *)(v62.m128i_i64[0] + 16) > 0x17u )
        {
          v69[0] = v62.m128i_i64[0];
          LODWORD(v68) = 1;
        }
        v58 = 0;
        v59 = 0;
        v60 = 0;
        v61 = 0;
        if ( !(unsigned __int8)sub_141CE30(a1, a2, v65.m128i_i64, &v62, v54 == 54, v14, a3, (__int64)&v58, 1u, v55) )
        {
          v40 = *(_DWORD *)(a3 + 12);
          *(_DWORD *)(a3 + 8) = 0;
          v56.m128i_i64[1] = 0x6000000000000003LL;
          v56.m128i_i64[0] = v14;
          v57 = v62.m128i_i64[0];
          v41 = 0;
          if ( !v40 )
          {
            sub_16CD150(a3, a3 + 16, 0, 24);
            v41 = 24LL * *(unsigned int *)(a3 + 8);
          }
          v42 = _mm_loadu_si128(&v56);
          v43 = (__m128i *)(*(_QWORD *)a3 + v41);
          v43[1].m128i_i64[0] = v57;
          *v43 = v42;
          ++*(_DWORD *)(a3 + 8);
        }
        j___libc_free_0(v59);
        if ( v67 != v69 )
          _libc_free((unsigned __int64)v67);
        return;
      }
    }
    else if ( v33 != 58 || (*(_BYTE *)(a2 + 18) & 1) == 0 )
    {
      goto LABEL_32;
    }
    v65.m128i_i64[0] = v14;
    v65.m128i_i64[1] = 0x6000000000000003LL;
    v66.m128i_i64[0] = v62.m128i_i64[0];
    v35 = *(unsigned int *)(a3 + 8);
    if ( (unsigned int)v35 >= *(_DWORD *)(a3 + 12) )
    {
      sub_16CD150(a3, a3 + 16, 0, 24);
      v35 = *(unsigned int *)(a3 + 8);
    }
    v36 = (__m128i *)(*(_QWORD *)a3 + 24 * v35);
    *v36 = _mm_loadu_si128(&v65);
    v36[1].m128i_i64[0] = v66.m128i_i64[0];
    ++*(_DWORD *)(a3 + 8);
  }
}

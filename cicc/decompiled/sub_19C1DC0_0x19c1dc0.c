// Function: sub_19C1DC0
// Address: 0x19c1dc0
//
__int64 __fastcall sub_19C1DC0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  const __m128i *v11; // r13
  __int64 v14; // rax
  __int64 result; // rax
  __int64 v16; // r12
  __int64 v17; // r8
  __int64 v18; // rax
  double v19; // xmm4_8
  double v20; // xmm5_8
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  int v25; // edi
  __int64 v26; // r10
  int v27; // edi
  unsigned int v28; // r8d
  __int64 *v29; // rcx
  __int64 v30; // r11
  _QWORD *v31; // rdx
  unsigned int v32; // r8d
  __int64 *v33; // rcx
  __int64 v34; // r11
  _QWORD *v35; // rax
  int v36; // ecx
  __int64 v37; // r15
  double v38; // xmm4_8
  double v39; // xmm5_8
  __int64 v40; // rax
  __int64 v41; // rdi
  __int64 *v42; // rsi
  __m128i *v43; // rdx
  __int64 v44; // r15
  __int64 v45; // r13
  __int64 v46; // rbx
  __int64 v47; // rax
  __int64 v48; // rax
  _BYTE *v49; // rsi
  __int64 *v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rax
  unsigned __int64 v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rcx
  __int64 v56; // rax
  __int64 v57; // rcx
  int v58; // r8d
  unsigned int v59; // edx
  __int64 v60; // rdi
  __int64 *v61; // r12
  char *v62; // rdx
  char *v63; // rdi
  __int64 v64; // rax
  __int64 v65; // rcx
  char *v66; // rax
  _QWORD *v67; // rax
  __int64 v68; // rax
  _QWORD *v69; // rdx
  __int64 v70; // rdx
  __int64 v71; // rcx
  int v72; // ecx
  int v73; // r9d
  int v74; // r9d
  __int64 *v75; // [rsp+10h] [rbp-80h]
  __int64 v76; // [rsp+10h] [rbp-80h]
  __int64 v77; // [rsp+18h] [rbp-78h]
  __m128i *v78; // [rsp+18h] [rbp-78h]
  __int64 *v79; // [rsp+18h] [rbp-78h]
  __int64 v81; // [rsp+28h] [rbp-68h]
  _QWORD v82[12]; // [rsp+30h] [rbp-60h] BYREF

  v11 = (const __m128i *)v82;
  v14 = sub_157EB90(**(_QWORD **)(a3 + 32));
  v81 = sub_1632FA0(v14);
  for ( result = *(_QWORD *)(a2 + 8); *(_QWORD *)a2 != result; result = *(_QWORD *)(a2 + 8) )
  {
    v16 = *(_QWORD *)(result - 8);
    *(_QWORD *)(a2 + 8) = result - 8;
    if ( (unsigned __int8)sub_1AE9990(v16, 0) )
    {
      if ( (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) != 0 )
      {
        v43 = (__m128i *)v11;
        v44 = 0;
        v45 = a1;
        v46 = 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
        do
        {
          if ( (*(_BYTE *)(v16 + 23) & 0x40) != 0 )
            v47 = *(_QWORD *)(v16 - 8);
          else
            v47 = v16 - 24LL * (*(_DWORD *)(v16 + 20) & 0xFFFFFFF);
          v48 = *(_QWORD *)(v47 + v44);
          if ( *(_BYTE *)(v48 + 16) > 0x17u )
          {
            v82[0] = v48;
            v49 = *(_BYTE **)(a2 + 8);
            if ( v49 == *(_BYTE **)(a2 + 16) )
            {
              v78 = v43;
              sub_170B610(a2, v49, v43);
              v43 = v78;
            }
            else
            {
              if ( v49 )
              {
                *(_QWORD *)v49 = v48;
                v49 = *(_BYTE **)(a2 + 8);
              }
              *(_QWORD *)(a2 + 8) = v49 + 8;
            }
          }
          v44 += 24;
        }
        while ( v46 != v44 );
        a1 = v45;
        v11 = v43;
      }
      sub_14045C0(*(_QWORD *)(a1 + 168), v16, a3);
      sub_19C03B0(v16, (char **)a2);
      sub_15F20C0((_QWORD *)v16);
      continue;
    }
    memset(&v82[1], 0, 32);
    v82[0] = v81;
    v18 = sub_13E3350(v16, v11, 0, 1, v17);
    v21 = v18;
    if ( v18 )
    {
      if ( *(_BYTE *)(v18 + 16) <= 0x17u )
        goto LABEL_13;
      v22 = *(_QWORD *)(v18 + 40);
      v23 = *(_QWORD *)(v16 + 40);
      if ( v22 == v23 )
        goto LABEL_13;
      v24 = *(_QWORD *)(a1 + 160);
      v25 = *(_DWORD *)(v24 + 24);
      if ( !v25 )
        goto LABEL_13;
      v26 = *(_QWORD *)(v24 + 8);
      v27 = v25 - 1;
      v28 = v27 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v29 = (__int64 *)(v26 + 16LL * v28);
      v30 = *v29;
      if ( v22 != *v29 )
      {
        v72 = 1;
        while ( v30 != -8 )
        {
          v73 = v72 + 1;
          v28 = v27 & (v72 + v28);
          v29 = (__int64 *)(v26 + 16LL * v28);
          v30 = *v29;
          if ( v22 == *v29 )
            goto LABEL_8;
          v72 = v73;
        }
LABEL_13:
        sub_19C1C30(v16, v21, a2, a3, *(_QWORD *)(a1 + 168), a4, a5, a6, a7, v19, v20, a10, a11);
        continue;
      }
LABEL_8:
      v31 = (_QWORD *)v29[1];
      if ( !v31 )
        goto LABEL_13;
      v32 = v27 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
      v33 = (__int64 *)(v26 + 16LL * v32);
      v34 = *v33;
      if ( v23 == *v33 )
      {
LABEL_10:
        v35 = (_QWORD *)v33[1];
        if ( v31 == v35 )
          goto LABEL_13;
        while ( v35 )
        {
          v35 = (_QWORD *)*v35;
          if ( v31 == v35 )
            goto LABEL_13;
        }
      }
      else
      {
        v36 = 1;
        while ( v34 != -8 )
        {
          v74 = v36 + 1;
          v32 = v27 & (v36 + v32);
          v33 = (__int64 *)(v26 + 16LL * v32);
          v34 = *v33;
          if ( v23 == *v33 )
            goto LABEL_10;
          v36 = v74;
        }
      }
    }
    if ( *(_BYTE *)(v16 + 16) == 26 && (*(_DWORD *)(v16 + 20) & 0xFFFFFFF) == 1 )
    {
      v37 = *(_QWORD *)(v16 - 24);
      v77 = *(_QWORD *)(v16 + 40);
      if ( sub_157F0B0(v37) )
      {
        while ( 1 )
        {
          v40 = *(_QWORD *)(v37 + 48);
          if ( !v40 )
            BUG();
          if ( *(_BYTE *)(v40 - 8) != 77 )
            break;
          if ( (*(_BYTE *)(v40 - 1) & 0x40) != 0 )
          {
            v42 = *(__int64 **)(v40 - 32);
            v41 = v40 - 24;
          }
          else
          {
            v41 = v40 - 24;
            v42 = (__int64 *)(v40 - 24 - 24LL * (*(_DWORD *)(v40 - 4) & 0xFFFFFFF));
          }
          sub_19C1C30(v41, *v42, a2, a3, *(_QWORD *)(a1 + 168), a4, a5, a6, a7, v38, v39, a10, a11);
        }
        sub_164D160(v37, v77, a4, a5, a6, a7, v38, v39, a10, a11);
        v50 = *(__int64 **)(v37 + 48);
        v51 = v37 + 40;
        if ( v50 != (__int64 *)(v37 + 40) )
        {
          v52 = v16 + 24;
          if ( v16 + 24 != v51 )
          {
            if ( v77 + 40 != v51 )
            {
              v75 = *(__int64 **)(v37 + 48);
              sub_157EA80(v77 + 40, v37 + 40, (__int64)v50, v51);
              v52 = v16 + 24;
              v50 = v75;
              v51 = v37 + 40;
            }
            if ( (__int64 *)v51 != v50 )
            {
              v53 = *(_QWORD *)(v37 + 40) & 0xFFFFFFFFFFFFFFF8LL;
              *(_QWORD *)((*v50 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v51;
              *(_QWORD *)(v37 + 40) = *(_QWORD *)(v37 + 40) & 7LL | *v50 & 0xFFFFFFFFFFFFFFF8LL;
              v54 = *(_QWORD *)(v16 + 24);
              *(_QWORD *)(v53 + 8) = v52;
              v54 &= 0xFFFFFFFFFFFFFFF8LL;
              *v50 = v54 | *v50 & 7;
              *(_QWORD *)(v54 + 8) = v50;
              *(_QWORD *)(v16 + 24) = v53 | *(_QWORD *)(v16 + 24) & 7LL;
            }
          }
        }
        sub_14045C0(*(_QWORD *)(a1 + 168), v16, a3);
        sub_19C03B0(v16, (char **)a2);
        sub_15F20C0((_QWORD *)v16);
        v55 = *(_QWORD *)(a1 + 160);
        v56 = *(unsigned int *)(v55 + 24);
        v76 = v55;
        if ( (_DWORD)v56 )
        {
          v57 = *(_QWORD *)(v55 + 8);
          v58 = 1;
          v59 = (v56 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v79 = (__int64 *)(v57 + 16LL * v59);
          v60 = *v79;
          if ( v37 == *v79 )
          {
LABEL_48:
            if ( v79 != (__int64 *)(v57 + 16 * v56) )
            {
              v61 = (__int64 *)v79[1];
              if ( v61 )
              {
                while ( 1 )
                {
                  v62 = (char *)v61[5];
                  v63 = (char *)v61[4];
                  v64 = (v62 - v63) >> 5;
                  v65 = (v62 - v63) >> 3;
                  if ( v64 <= 0 )
                    break;
                  v66 = &v63[32 * v64];
                  while ( v37 != *(_QWORD *)v63 )
                  {
                    if ( v37 == *((_QWORD *)v63 + 1) )
                    {
                      v63 += 8;
                      break;
                    }
                    if ( v37 == *((_QWORD *)v63 + 2) )
                    {
                      v63 += 16;
                      break;
                    }
                    if ( v37 == *((_QWORD *)v63 + 3) )
                    {
                      v63 += 24;
                      break;
                    }
                    v63 += 32;
                    if ( v66 == v63 )
                    {
                      v65 = (v62 - v63) >> 3;
                      goto LABEL_77;
                    }
                  }
LABEL_57:
                  if ( v63 + 8 != v62 )
                  {
                    memmove(v63, v63 + 8, v62 - (v63 + 8));
                    v62 = (char *)v61[5];
                  }
                  v67 = (_QWORD *)v61[8];
                  v61[5] = (__int64)(v62 - 8);
                  if ( (_QWORD *)v61[9] == v67 )
                  {
                    v69 = &v67[*((unsigned int *)v61 + 21)];
                    if ( v67 == v69 )
                    {
LABEL_75:
                      v67 = v69;
                    }
                    else
                    {
                      while ( v37 != *v67 )
                      {
                        if ( v69 == ++v67 )
                          goto LABEL_75;
                      }
                    }
                    goto LABEL_70;
                  }
                  v67 = sub_16CC9F0((__int64)(v61 + 7), v37);
                  if ( v37 == *v67 )
                  {
                    v70 = v61[9];
                    if ( v70 == v61[8] )
                      v71 = *((unsigned int *)v61 + 21);
                    else
                      v71 = *((unsigned int *)v61 + 20);
                    v69 = (_QWORD *)(v70 + 8 * v71);
LABEL_70:
                    if ( v67 != v69 )
                    {
                      *v67 = -2;
                      ++*((_DWORD *)v61 + 22);
                    }
                    goto LABEL_62;
                  }
                  v68 = v61[9];
                  if ( v68 == v61[8] )
                  {
                    v67 = (_QWORD *)(v68 + 8LL * *((unsigned int *)v61 + 21));
                    v69 = v67;
                    goto LABEL_70;
                  }
LABEL_62:
                  v61 = (__int64 *)*v61;
                  if ( !v61 )
                    goto LABEL_63;
                }
LABEL_77:
                if ( v65 != 2 )
                {
                  if ( v65 != 3 )
                  {
                    if ( v65 != 1 )
                    {
                      v63 = (char *)v61[5];
                      goto LABEL_57;
                    }
LABEL_89:
                    if ( v37 != *(_QWORD *)v63 )
                      v63 = (char *)v61[5];
                    goto LABEL_57;
                  }
                  if ( v37 == *(_QWORD *)v63 )
                    goto LABEL_57;
                  v63 += 8;
                }
                if ( v37 == *(_QWORD *)v63 )
                  goto LABEL_57;
                v63 += 8;
                goto LABEL_89;
              }
LABEL_63:
              *v79 = -16;
              --*(_DWORD *)(v76 + 16);
              ++*(_DWORD *)(v76 + 20);
            }
          }
          else
          {
            while ( v60 != -8 )
            {
              v59 = (v56 - 1) & (v58 + v59);
              v79 = (__int64 *)(v57 + 16LL * v59);
              v60 = *v79;
              if ( v37 == *v79 )
                goto LABEL_48;
              ++v58;
            }
          }
        }
        sub_14045C0(*(_QWORD *)(a1 + 168), v37, a3);
        sub_157F980(v37);
      }
    }
  }
  return result;
}

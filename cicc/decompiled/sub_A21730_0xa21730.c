// Function: sub_A21730
// Address: 0xa21730
//
__int64 __fastcall sub_A21730(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 result; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rbx
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rbx
  __int64 v14; // r15
  int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rbx
  unsigned int v18; // r8d
  unsigned int v19; // r10d
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rax
  unsigned __int64 v25; // rdx
  int v26; // edx
  __int64 v27; // rsi
  __int64 v28; // rdi
  int v29; // edx
  unsigned int v30; // r8d
  __int64 *v31; // rcx
  __int64 v32; // r10
  int v33; // r8d
  __int64 v34; // rdx
  __int64 v35; // rdi
  __int64 v36; // r8
  __int64 v37; // rsi
  unsigned int v38; // ecx
  __int64 *v39; // rdx
  __int64 v40; // r9
  int v41; // r8d
  int v42; // edx
  int v43; // ecx
  int v44; // r11d
  int v45; // r9d
  __int64 v46; // [rsp+0h] [rbp-180h]
  __int64 v47; // [rsp+8h] [rbp-178h]
  unsigned int v48; // [rsp+10h] [rbp-170h]
  int v49; // [rsp+14h] [rbp-16Ch]
  unsigned int v50; // [rsp+14h] [rbp-16Ch]
  int v51; // [rsp+14h] [rbp-16Ch]
  __int64 v52; // [rsp+20h] [rbp-160h]
  __int64 v53; // [rsp+30h] [rbp-150h]
  _BYTE *v55; // [rsp+40h] [rbp-140h] BYREF
  __int64 v56; // [rsp+48h] [rbp-138h]
  _BYTE v57[304]; // [rsp+50h] [rbp-130h] BYREF

  v3 = a3;
  v55 = v57;
  v52 = a2;
  v56 = 0x4000000000LL;
  result = sub_BD5C60(a2, a2, a3);
  v47 = result;
  if ( *(char *)(a2 + 7) < 0 )
  {
    v5 = sub_BD2BC0(a2);
    v7 = v5 + v6;
    if ( *(char *)(a2 + 7) < 0 )
      v7 -= sub_BD2BC0(a2);
    v53 = 0;
    v8 = v7 >> 4;
    v46 = 16LL * (unsigned int)v8;
    result = a1;
    v9 = a1 + 24;
    if ( (_DWORD)v8 )
    {
      while ( 1 )
      {
        v10 = 0;
        if ( *(char *)(v52 + 7) < 0 )
          v10 = sub_BD2BC0(v52);
        v11 = v53 + v10;
        v12 = 32LL * *(unsigned int *)(v11 + 8);
        v13 = 32LL * *(unsigned int *)(v11 + 12) - v12;
        v14 = v52 + v12 - 32LL * (*(_DWORD *)(v52 + 4) & 0x7FFFFFF);
        v15 = sub_B6F800(v47, *(_QWORD *)v11 + 16LL, **(_QWORD **)v11);
        v16 = (unsigned int)v56;
        if ( (unsigned __int64)(unsigned int)v56 + 1 > HIDWORD(v56) )
        {
          v51 = v15;
          sub_C8D5F0(&v55, v57, (unsigned int)v56 + 1LL, 4);
          v16 = (unsigned int)v56;
          v15 = v51;
        }
        v17 = v14 + v13;
        *(_DWORD *)&v55[4 * v16] = v15;
        LODWORD(v56) = v56 + 1;
        if ( v17 != v14 )
          break;
LABEL_23:
        a2 = 55;
        sub_A214F0(*(_QWORD *)a1, 0x37u, (__int64)&v55, 0);
        v53 += 16;
        result = v53;
        LODWORD(v56) = 0;
        if ( v46 == v53 )
          goto LABEL_24;
      }
      while ( 1 )
      {
        while ( 1 )
        {
          v23 = *(_QWORD *)v14;
          if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)v14 + 8LL) + 8LL) == 9 )
          {
            v24 = (unsigned int)v56;
            v25 = (unsigned int)v56 + 1LL;
            if ( v25 > HIDWORD(v56) )
            {
              sub_C8D5F0(&v55, v57, v25, 4);
              v24 = (unsigned int)v56;
            }
            *(_DWORD *)&v55[4 * v24] = 0x80000000;
            v26 = *(_DWORD *)(a1 + 304);
            v27 = *(_QWORD *)(v23 + 24);
            v22 = (unsigned int)(v56 + 1);
            v28 = *(_QWORD *)(a1 + 288);
            LODWORD(v56) = v56 + 1;
            if ( v26 )
            {
              v29 = v26 - 1;
              v30 = v29 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
              v31 = (__int64 *)(v28 + 16LL * v30);
              v32 = *v31;
              if ( v27 == *v31 )
              {
LABEL_20:
                v26 = *((_DWORD *)v31 + 3);
              }
              else
              {
                v43 = 1;
                while ( v32 != -4096 )
                {
                  v45 = v43 + 1;
                  v30 = v29 & (v43 + v30);
                  v31 = (__int64 *)(v28 + 16LL * v30);
                  v32 = *v31;
                  if ( v27 == *v31 )
                    goto LABEL_20;
                  v43 = v45;
                }
                v26 = 0;
              }
            }
            v33 = v3 + 1 - v26;
            v34 = v22 + 1;
            if ( v22 + 1 <= (unsigned __int64)HIDWORD(v56) )
              goto LABEL_22;
LABEL_30:
            v49 = v33;
            sub_C8D5F0(&v55, v57, v34, 4);
            v22 = (unsigned int)v56;
            v33 = v49;
            goto LABEL_22;
          }
          v18 = sub_A3F3B0(v9);
          v19 = v3 - v18;
          v20 = (unsigned int)v56;
          v21 = (unsigned int)v56 + 1LL;
          if ( v21 > HIDWORD(v56) )
          {
            v48 = v3 - v18;
            v50 = v18;
            sub_C8D5F0(&v55, v57, v21, 4);
            v20 = (unsigned int)v56;
            v19 = v48;
            v18 = v50;
          }
          *(_DWORD *)&v55[4 * v20] = v19;
          v22 = (unsigned int)(v56 + 1);
          LODWORD(v56) = v56 + 1;
          if ( v3 <= v18 )
            break;
          v14 += 32;
          if ( v17 == v14 )
            goto LABEL_23;
        }
        v35 = *(_QWORD *)(v23 + 8);
        v36 = *(_QWORD *)(a1 + 56);
        v37 = *(unsigned int *)(a1 + 72);
        if ( (_DWORD)v37 )
        {
          v38 = (v37 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
          v39 = (__int64 *)(v36 + 16LL * v38);
          v40 = *v39;
          if ( *v39 == v35 )
            goto LABEL_29;
          v42 = 1;
          while ( v40 != -4096 )
          {
            v44 = v42 + 1;
            v38 = (v37 - 1) & (v42 + v38);
            v39 = (__int64 *)(v36 + 16LL * v38);
            v40 = *v39;
            if ( v35 == *v39 )
              goto LABEL_29;
            v42 = v44;
          }
        }
        v39 = (__int64 *)(v36 + 16 * v37);
LABEL_29:
        v41 = *((_DWORD *)v39 + 2);
        v34 = v22 + 1;
        v33 = v41 - 1;
        if ( v22 + 1 > (unsigned __int64)HIDWORD(v56) )
          goto LABEL_30;
LABEL_22:
        v14 += 32;
        *(_DWORD *)&v55[4 * v22] = v33;
        LODWORD(v56) = v56 + 1;
        if ( v17 == v14 )
          goto LABEL_23;
      }
    }
  }
LABEL_24:
  if ( v55 != v57 )
    return _libc_free(v55, a2);
  return result;
}

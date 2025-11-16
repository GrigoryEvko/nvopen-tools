// Function: sub_CFC6F0
// Address: 0xcfc6f0
//
unsigned __int8 *__fastcall sub_CFC6F0(
        __int64 a1,
        unsigned __int8 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6)
{
  unsigned int v7; // r12d
  bool v8; // sf
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rbx
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 *v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rdx
  unsigned __int8 **v17; // rbx
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r9
  int v25; // esi
  __int64 v26; // rdi
  __int64 *v27; // r9
  __int64 v28; // rax
  __int64 v29; // rsi
  unsigned __int8 *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // r9
  int v34; // esi
  __int64 v35; // rbx
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 i; // rax
  __int64 v39; // r12
  unsigned __int8 *result; // rax
  unsigned __int8 *v41; // rax
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned __int64 v45; // rsi
  unsigned __int64 v46; // rsi
  __int64 v47; // [rsp+0h] [rbp-80h]
  __int64 *v48; // [rsp+8h] [rbp-78h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  __int64 v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+8h] [rbp-78h]
  __int64 v52; // [rsp+10h] [rbp-70h]
  __int64 v53; // [rsp+10h] [rbp-70h]
  __int64 v54; // [rsp+10h] [rbp-70h]
  __int64 *v55; // [rsp+10h] [rbp-70h]
  __int64 v56; // [rsp+10h] [rbp-70h]
  __int64 v57; // [rsp+10h] [rbp-70h]
  __int64 v58; // [rsp+10h] [rbp-70h]
  __int64 v59; // [rsp+10h] [rbp-70h]
  __int64 v61; // [rsp+20h] [rbp-60h] BYREF
  __int64 v62; // [rsp+28h] [rbp-58h] BYREF
  __int64 v63; // [rsp+30h] [rbp-50h] BYREF
  __int64 v64; // [rsp+38h] [rbp-48h]
  unsigned __int8 *v65; // [rsp+40h] [rbp-40h]
  unsigned int v66; // [rsp+48h] [rbp-38h]

  v7 = 0;
  v8 = *(char *)(a1 + 7) < 0;
  v61 = a3;
  v62 = a3;
  if ( !v8 )
    goto LABEL_39;
LABEL_2:
  v9 = sub_BD2BC0(a1);
  v11 = v9 + v10;
  if ( *(char *)(a1 + 7) >= 0 )
  {
    for ( i = v11 >> 4; ; LODWORD(i) = 0 )
    {
      if ( v7 == (_DWORD)i )
        goto LABEL_46;
LABEL_41:
      v12 = 0;
LABEL_6:
      v13 = v12 + 16LL * v7;
      v14 = *(__int64 **)v13;
      v15 = 32LL * *(unsigned int *)(v13 + 8);
      v16 = **(_QWORD **)v13;
      v17 = (unsigned __int8 **)(a1 + v15 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
      v18 = (32LL * *(unsigned int *)(v13 + 12) - v15) >> 5;
      if ( *v14 == 16 )
      {
        a4 = v14[3] ^ 0x656761726F74735FLL;
        if ( v14[2] ^ 0x6574617261706573LL | a4 )
        {
          if ( v18 )
            goto LABEL_45;
        }
        else
        {
          v19 = sub_98ACB0(*v17, 6u);
          if ( (unsigned __int8)(*v19 - 4) > 0x18u || *v19 == 22 )
          {
            v63 = 4;
            v22 = v62;
            v64 = 0;
            v65 = v19;
            if ( v19 != (unsigned __int8 *)-8192LL && v19 != (unsigned __int8 *)-4096LL )
            {
              v52 = v62;
              sub_BD73F0((__int64)&v63);
              v22 = v52;
            }
            v66 = v7;
            v23 = *(unsigned int *)(v22 + 8);
            v24 = v23 + 1;
            v25 = *(_DWORD *)(v22 + 8);
            if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v22 + 12) )
            {
              v46 = v23 + 1;
              if ( *(_QWORD *)v22 > (unsigned __int64)&v63
                || (v51 = *(_QWORD *)v22, (unsigned __int64)&v63 >= *(_QWORD *)v22 + 32 * v23) )
              {
                v58 = v22;
                sub_CFC2E0(v22, v46, v22, v20, v21, v24);
                v22 = v58;
                v27 = &v63;
                v23 = *(unsigned int *)(v58 + 8);
                v26 = *(_QWORD *)v58;
                v25 = *(_DWORD *)(v58 + 8);
              }
              else
              {
                v57 = v22;
                sub_CFC2E0(v22, v46, v22, v20, v21, v24);
                v22 = v57;
                v26 = *(_QWORD *)v57;
                v23 = *(unsigned int *)(v57 + 8);
                v27 = (__int64 *)((char *)&v63 + *(_QWORD *)v57 - v51);
                v25 = *(_DWORD *)(v57 + 8);
              }
            }
            else
            {
              v26 = *(_QWORD *)v22;
              v27 = &v63;
            }
            v28 = v26 + 32 * v23;
            if ( v28 )
            {
              *(_QWORD *)v28 = 4;
              v29 = v27[2];
              *(_QWORD *)(v28 + 8) = 0;
              *(_QWORD *)(v28 + 16) = v29;
              if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
              {
                v47 = v22;
                v48 = v27;
                v53 = v28;
                sub_BD6050((unsigned __int64 *)v28, *v27 & 0xFFFFFFFFFFFFFFF8LL);
                v22 = v47;
                v27 = v48;
                v28 = v53;
              }
              *(_DWORD *)(v28 + 24) = *((_DWORD *)v27 + 6);
              v25 = *(_DWORD *)(v22 + 8);
            }
            *(_DWORD *)(v22 + 8) = v25 + 1;
            if ( v65 != 0 && v65 + 4096 != 0 && v65 != (unsigned __int8 *)-8192LL )
              sub_BD60C0(&v63);
          }
          v30 = sub_98ACB0(v17[4], 6u);
          if ( (unsigned __int8)(*v30 - 4) > 0x18u || *v30 == 22 )
          {
            v63 = 4;
            v31 = v62;
            v64 = 0;
            v65 = v30;
            if ( v30 != (unsigned __int8 *)-4096LL && v30 != (unsigned __int8 *)-8192LL )
            {
              v54 = v62;
              sub_BD73F0((__int64)&v63);
              v31 = v54;
            }
            v66 = v7;
            v32 = *(unsigned int *)(v31 + 8);
            v33 = v32 + 1;
            v34 = *(_DWORD *)(v31 + 8);
            if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 12) )
            {
              v45 = v32 + 1;
              if ( *(_QWORD *)v31 > (unsigned __int64)&v63
                || (v50 = *(_QWORD *)v31, (unsigned __int64)&v63 >= *(_QWORD *)v31 + 32 * v32) )
              {
                v59 = v31;
                sub_CFC2E0(v31, v45, v31, a4, a5, v33);
                v31 = v59;
                a6 = &v63;
                v32 = *(unsigned int *)(v59 + 8);
                v35 = *(_QWORD *)v59;
                v34 = *(_DWORD *)(v59 + 8);
              }
              else
              {
                v56 = v31;
                sub_CFC2E0(v31, v45, v31, a4, a5, v33);
                v31 = v56;
                v35 = *(_QWORD *)v56;
                v32 = *(unsigned int *)(v56 + 8);
                a6 = (__int64 *)((char *)&v63 + *(_QWORD *)v56 - v50);
                v34 = *(_DWORD *)(v56 + 8);
              }
            }
            else
            {
              v35 = *(_QWORD *)v31;
              a6 = &v63;
            }
            v36 = 32 * v32 + v35;
            if ( v36 )
            {
              *(_QWORD *)v36 = 4;
              v37 = a6[2];
              *(_QWORD *)(v36 + 8) = 0;
              *(_QWORD *)(v36 + 16) = v37;
              if ( v37 != -4096 && v37 != 0 && v37 != -8192 )
              {
                v49 = v31;
                v55 = a6;
                sub_BD6050((unsigned __int64 *)v36, *a6 & 0xFFFFFFFFFFFFFFF8LL);
                v31 = v49;
                a6 = v55;
              }
              *(_DWORD *)(v36 + 24) = *((_DWORD *)a6 + 6);
              v34 = *(_DWORD *)(v31 + 8);
            }
            *(_DWORD *)(v31 + 8) = v34 + 1;
            if ( v65 + 4096 != 0 && v65 != 0 && v65 != (unsigned __int8 *)-8192LL )
              sub_BD60C0(&v63);
          }
        }
      }
      else if ( v18 && (v16 != 6 || *((_DWORD *)v14 + 4) != 1869506409 || *((_WORD *)v14 + 10) != 25970) )
      {
LABEL_45:
        sub_CFC580(&v62, *v17, v7, a4, a5, (__int64)a6);
      }
      ++v7;
      if ( *(char *)(a1 + 7) < 0 )
        goto LABEL_2;
LABEL_39:
      ;
    }
  }
  if ( v7 != (unsigned int)((v11 - sub_BD2BC0(a1)) >> 4) )
  {
    if ( *(char *)(a1 + 7) < 0 )
    {
      v12 = sub_BD2BC0(a1);
      goto LABEL_6;
    }
    goto LABEL_41;
  }
LABEL_46:
  v39 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  sub_997910(v39, 1, (__int64 (__fastcall *)(__int64, unsigned __int8 *))sub_CFC410, (__int64)&v61);
  result = a2;
  if ( a2 )
  {
    result = (unsigned __int8 *)sub_DF9BD0(a2, v39);
    if ( result )
    {
      v41 = sub_BD4CB0(result, (void (__fastcall *)(__int64, unsigned __int8 *))nullsub_182, (__int64)&v63);
      return (unsigned __int8 *)sub_CFC580(&v62, v41, -1, v42, v43, v44);
    }
  }
  return result;
}

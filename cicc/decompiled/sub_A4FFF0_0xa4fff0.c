// Function: sub_A4FFF0
// Address: 0xa4fff0
//
void __fastcall sub_A4FFF0(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned int *a8)
{
  __int64 v8; // rsi
  __int64 v10; // rbx
  __int64 v11; // rsi
  __int64 v12; // rbx
  bool v13; // al
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  int v17; // eax
  __int64 v18; // rsi
  __int64 v19; // rdx
  unsigned __int64 v20; // r14
  __int64 *v21; // r12
  __int64 v22; // rax
  int v23; // edx
  char *v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // r12
  __int64 v30; // rax
  int v31; // r10d
  unsigned int v32; // ecx
  __int64 *v33; // rdx
  __int64 v34; // r11
  __int64 *v35; // rax
  unsigned int v36; // r11d
  unsigned int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r15
  unsigned int v40; // eax
  char v41; // dl
  unsigned int v42; // r15d
  bool v43; // al
  int v44; // edx
  int v45; // edx
  unsigned int v46; // r15d
  int v47; // eax
  int v48; // ecx
  int v49; // r15d
  __int64 v50; // rbx
  __int128 v51; // xmm0
  __int64 v52; // r12
  __int64 v53; // r9
  __int64 *v54; // rbx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // rax
  int v58; // edx
  bool v59; // zf
  int v60; // edx
  int v61; // eax
  int v62; // [rsp+Ch] [rbp-B4h]
  __int64 v63; // [rsp+10h] [rbp-B0h]
  __int64 v64; // [rsp+18h] [rbp-A8h]
  unsigned int *v65; // [rsp+20h] [rbp-A0h]
  __int64 v66; // [rsp+28h] [rbp-98h]
  __int64 v67; // [rsp+30h] [rbp-90h]
  __int64 *v68; // [rsp+38h] [rbp-88h]
  __int64 *v69; // [rsp+40h] [rbp-80h]
  unsigned int *v70; // [rsp+60h] [rbp-60h]
  __int128 v71; // [rsp+70h] [rbp-50h] BYREF
  unsigned int *v72; // [rsp+80h] [rbp-40h]

  v68 = a2;
  v8 = (__int64)a2 - a1;
  v67 = a3;
  if ( v8 <= 256 )
    return;
  v10 = v8;
  if ( !a3 )
  {
    v69 = v68;
    goto LABEL_53;
  }
  v64 = a7;
  v63 = a1 + 16;
  v65 = a8;
  v66 = *((_QWORD *)&a7 + 1);
  while ( 2 )
  {
    v11 = *(_QWORD *)(a1 + 16);
    --v67;
    *(_QWORD *)&v71 = v64;
    v12 = a1 + 16 * (v10 >> 5);
    *((_QWORD *)&v71 + 1) = v66;
    v72 = v65;
    v13 = sub_A4FAF0((__int64 *)&v71, v11, *(_QWORD *)v12);
    v14 = *(v68 - 2);
    if ( !v13 )
    {
      if ( !sub_A4FAF0((__int64 *)&v71, *(_QWORD *)(a1 + 16), v14) )
      {
        v59 = !sub_A4FAF0((__int64 *)&v71, *(_QWORD *)v12, *(v68 - 2));
        v15 = *(_QWORD *)a1;
        if ( !v59 )
        {
          *(_QWORD *)a1 = *(v68 - 2);
          v60 = *((_DWORD *)v68 - 2);
          *(v68 - 2) = v15;
          v61 = *(_DWORD *)(a1 + 8);
          *(_DWORD *)(a1 + 8) = v60;
          *((_DWORD *)v68 - 2) = v61;
          v18 = *(_QWORD *)(a1 + 16);
          v19 = *(_QWORD *)a1;
          goto LABEL_8;
        }
        goto LABEL_7;
      }
LABEL_49:
      v18 = *(_QWORD *)a1;
      v19 = *(_QWORD *)(a1 + 16);
      v47 = *(_DWORD *)(a1 + 8);
      v48 = *(_DWORD *)(a1 + 24);
      *(_QWORD *)a1 = v19;
      *(_QWORD *)(a1 + 16) = v18;
      *(_DWORD *)(a1 + 8) = v48;
      *(_DWORD *)(a1 + 24) = v47;
      goto LABEL_8;
    }
    if ( !sub_A4FAF0((__int64 *)&v71, *(_QWORD *)v12, v14) )
    {
      if ( sub_A4FAF0((__int64 *)&v71, *(_QWORD *)(a1 + 16), *(v68 - 2)) )
      {
        v57 = *(_QWORD *)a1;
        *(_QWORD *)a1 = *(v68 - 2);
        v58 = *((_DWORD *)v68 - 2);
        *(v68 - 2) = v57;
        LODWORD(v57) = *(_DWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 8) = v58;
        *((_DWORD *)v68 - 2) = v57;
        v18 = *(_QWORD *)(a1 + 16);
        v19 = *(_QWORD *)a1;
        goto LABEL_8;
      }
      goto LABEL_49;
    }
    v15 = *(_QWORD *)a1;
LABEL_7:
    *(_QWORD *)a1 = *(_QWORD *)v12;
    v16 = *(_DWORD *)(v12 + 8);
    *(_QWORD *)v12 = v15;
    v17 = *(_DWORD *)(a1 + 8);
    *(_DWORD *)(a1 + 8) = v16;
    *(_DWORD *)(v12 + 8) = v17;
    v18 = *(_QWORD *)(a1 + 16);
    v19 = *(_QWORD *)a1;
LABEL_8:
    v20 = v63;
    v21 = v68;
    *(_QWORD *)&v71 = v64;
    *((_QWORD *)&v71 + 1) = v66;
    v72 = v65;
    while ( 1 )
    {
      v69 = (__int64 *)v20;
      if ( !sub_A4FAF0((__int64 *)&v71, v18, v19) )
        break;
LABEL_13:
      v19 = *(_QWORD *)a1;
      v18 = *(_QWORD *)(v20 + 16);
      v20 += 16LL;
    }
    v27 = (unsigned __int64)(v21 - 2);
    v28 = *(_QWORD *)a1;
    v29 = *(v21 - 2);
    if ( v29 != *(_QWORD *)a1 )
    {
      do
      {
        v26 = *(_QWORD *)(v28 + 24);
        v30 = *(unsigned int *)(v71 + 24);
        v25 = *(_QWORD *)(v71 + 8);
        if ( !(_DWORD)v30 )
        {
          v36 = 0;
LABEL_29:
          v24 = (char *)*((_QWORD *)&v71 + 1);
LABEL_30:
          v41 = *v24;
          if ( v36 )
          {
LABEL_24:
            if ( v41 && *v72 >= v36 )
              break;
            goto LABEL_26;
          }
          goto LABEL_31;
        }
        v31 = v30 - 1;
        v32 = (v30 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v33 = (__int64 *)(v25 + 16LL * v32);
        v34 = *v33;
        if ( v26 == *v33 )
        {
LABEL_18:
          v35 = (__int64 *)(v25 + 16 * v30);
          if ( v35 != v33 )
          {
            v26 = *(_QWORD *)(v29 + 24);
            v36 = *(_DWORD *)(*(_QWORD *)(v71 + 32) + 16LL * *((unsigned int *)v33 + 2) + 8);
            goto LABEL_20;
          }
        }
        else
        {
          v44 = 1;
          while ( v34 != -4096 )
          {
            v49 = v44 + 1;
            v32 = v31 & (v44 + v32);
            v33 = (__int64 *)(v25 + 16LL * v32);
            v34 = *v33;
            if ( v26 == *v33 )
              goto LABEL_18;
            v44 = v49;
          }
          v35 = (__int64 *)(v25 + 16 * v30);
        }
        v26 = *(_QWORD *)(v29 + 24);
        v36 = 0;
LABEL_20:
        v37 = v31 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v38 = (__int64 *)(v25 + 16LL * v37);
        v39 = *v38;
        if ( v26 != *v38 )
        {
          v45 = 1;
          while ( v39 != -4096 )
          {
            v37 = v31 & (v45 + v37);
            v62 = v45 + 1;
            v38 = (__int64 *)(v25 + 16LL * v37);
            v39 = *v38;
            if ( v26 == *v38 )
              goto LABEL_21;
            v45 = v62;
          }
          goto LABEL_29;
        }
LABEL_21:
        v24 = (char *)*((_QWORD *)&v71 + 1);
        LODWORD(v25) = (unsigned __int8)**((_BYTE **)&v71 + 1);
        if ( v35 == v38 )
          goto LABEL_30;
        v40 = *(_DWORD *)(*(_QWORD *)(v71 + 32) + 16LL * *((unsigned int *)v38 + 2) + 8);
        if ( v40 > v36 )
        {
          if ( !(_BYTE)v25 || *v72 < v40 )
            break;
          goto LABEL_26;
        }
        v41 = **((_BYTE **)&v71 + 1);
        if ( v40 < v36 )
          goto LABEL_24;
LABEL_31:
        if ( v41 && *v72 >= v36 )
        {
          v46 = sub_BD2910(v28);
          v43 = v46 < (unsigned int)sub_BD2910(v29);
        }
        else
        {
          v42 = sub_BD2910(v28);
          v43 = v42 > (unsigned int)sub_BD2910(v29);
        }
        if ( !v43 )
          break;
        v28 = *(_QWORD *)a1;
LABEL_26:
        v29 = *(_QWORD *)(v27 - 16);
        v27 -= 16LL;
      }
      while ( v28 != v29 );
    }
    if ( v20 < v27 )
    {
      v22 = *(_QWORD *)v20;
      v21 = (__int64 *)v27;
      *(_QWORD *)v20 = *(_QWORD *)v27;
      v23 = *(_DWORD *)(v27 + 8);
      *(_QWORD *)v27 = v22;
      LODWORD(v22) = *(_DWORD *)(v20 + 8);
      *(_DWORD *)(v20 + 8) = v23;
      *(_DWORD *)(v27 + 8) = v22;
      goto LABEL_13;
    }
    v10 = v20 - a1;
    sub_A4FFF0(v20, (_DWORD)v68, v67, (_DWORD)v24, v25, v26, a7, (__int64)a8);
    if ( (__int64)(v20 - a1) > 256 )
    {
      if ( v67 )
      {
        v68 = (__int64 *)v20;
        continue;
      }
LABEL_53:
      v50 = v10 >> 4;
      v51 = (__int128)_mm_loadu_si128((const __m128i *)&a7);
      v52 = (v50 - 2) >> 1;
      v72 = a8;
      v70 = a8;
      v71 = v51;
      sub_A4FE10(a1, v52, v50, *(_QWORD *)(a1 + 16 * v52), *(_QWORD *)(a1 + 16 * v52 + 8), a6, v51, (__int64)a8);
      do
      {
        --v52;
        sub_A4FE10(a1, v52, v50, *(_QWORD *)(a1 + 16 * v52), *(_QWORD *)(a1 + 16 * v52 + 8), v53, v71, (__int64)v72);
      }
      while ( v52 );
      v54 = v69;
      do
      {
        v54 -= 2;
        v55 = *v54;
        v56 = v54[1];
        *v54 = *(_QWORD *)a1;
        *((_DWORD *)v54 + 2) = *(_DWORD *)(a1 + 8);
        sub_A4FE10(a1, 0, ((__int64)v54 - a1) >> 4, v55, v56, v53, v51, (__int64)v70);
      }
      while ( (__int64)v54 - a1 > 16 );
    }
    break;
  }
}

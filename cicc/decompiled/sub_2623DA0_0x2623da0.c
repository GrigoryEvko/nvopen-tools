// Function: sub_2623DA0
// Address: 0x2623da0
//
void __fastcall sub_2623DA0(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  int *v8; // r13
  __int64 v9; // rbx
  __int64 v10; // r10
  __int64 v11; // r12
  __int64 v12; // r15
  int *v13; // rbx
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // r14
  unsigned __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rbx
  int *v22; // r15
  __int64 i; // r12
  unsigned __int64 v24; // r14
  __int64 v25; // r13
  unsigned __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // r15
  __int64 v29; // r13
  __int64 v30; // rbx
  unsigned __int64 v31; // r14
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // r12
  __int64 v38; // r13
  __int64 v39; // rbx
  unsigned __int64 v40; // r15
  unsigned __int64 v41; // r14
  unsigned __int64 v42; // rdi
  __int64 v43; // rdx
  __int64 v44; // rdx
  __int64 v45; // rbx
  __int64 v46; // r15
  int *v47; // r14
  unsigned __int64 v48; // r13
  unsigned __int64 v49; // rdi
  __int64 v50; // rdx
  __int64 v51; // rdx
  unsigned __int64 v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  int *v56; // r14
  __int64 v57; // rbx
  unsigned __int64 v58; // r13
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rdi
  __int64 v61; // rdx
  __int64 v62; // rdx
  __int64 v63; // r12
  int *v64; // r12
  unsigned __int64 v65; // r14
  int *v66; // rbx
  unsigned __int64 v67; // r13
  unsigned __int64 v68; // rdi
  __int64 v69; // rcx
  __int64 v70; // rcx
  unsigned __int64 v71; // rdi
  __int64 v72; // rcx
  unsigned __int64 v73; // r12
  __int64 v74; // r14
  __int64 v75; // r13
  __int64 v76; // r14
  unsigned __int64 v77; // rbx
  unsigned __int64 v78; // rdi
  __int64 v79; // rax
  __int64 v80; // rax
  unsigned __int64 v81; // r14
  __int64 v82; // r13
  __int64 v83; // r12
  unsigned __int64 v84; // rbx
  unsigned __int64 v85; // rdi
  __int64 v86; // rax
  __int64 v87; // rax
  __int64 v88; // [rsp+8h] [rbp-88h]
  __int64 v89; // [rsp+10h] [rbp-80h]
  __int64 v90; // [rsp+18h] [rbp-78h]
  __int64 v91; // [rsp+18h] [rbp-78h]
  __int64 v92; // [rsp+18h] [rbp-78h]
  __int64 v93; // [rsp+20h] [rbp-70h]
  int *v94; // [rsp+28h] [rbp-68h]
  __int64 v95; // [rsp+30h] [rbp-60h]
  __int64 v96; // [rsp+38h] [rbp-58h]
  int *v97; // [rsp+40h] [rbp-50h]
  __int64 v98; // [rsp+40h] [rbp-50h]
  __int64 v99; // [rsp+40h] [rbp-50h]
  __int64 v100; // [rsp+48h] [rbp-48h]
  __int64 v101; // [rsp+48h] [rbp-48h]
  __int64 v102; // [rsp+48h] [rbp-48h]
  __int64 v103; // [rsp+50h] [rbp-40h]
  __int64 v104; // [rsp+58h] [rbp-38h]

  while ( 1 )
  {
    v7 = a5;
    v8 = a2;
    v9 = a6;
    v103 = a1;
    v104 = a3;
    v100 = a5;
    if ( a5 > a7 )
      v7 = a7;
    v97 = a2;
    if ( v7 >= a4 )
      break;
    if ( a5 <= a7 )
    {
      v101 = a3 - (_QWORD)a2;
      if ( a3 - (__int64)a2 <= 0 )
        return;
      v99 = a6;
      v12 = a6 + 8;
      v13 = a2 + 2;
      v14 = 0xCCCCCCCCCCCCCCCDLL * ((a3 - (__int64)a2) >> 4);
      do
      {
        v15 = *(_QWORD *)(v12 + 8);
        while ( v15 )
        {
          sub_261DCB0(*(_QWORD *)(v15 + 24));
          v16 = v15;
          v15 = *(_QWORD *)(v15 + 16);
          j_j___libc_free_0(v16);
        }
        *(_QWORD *)(v12 + 8) = 0;
        *(_QWORD *)(v12 + 16) = v12;
        *(_QWORD *)(v12 + 24) = v12;
        *(_QWORD *)(v12 + 32) = 0;
        if ( *((_QWORD *)v13 + 1) )
        {
          *(_DWORD *)v12 = *v13;
          v17 = *((_QWORD *)v13 + 1);
          *(_QWORD *)(v12 + 8) = v17;
          *(_QWORD *)(v12 + 16) = *((_QWORD *)v13 + 2);
          *(_QWORD *)(v12 + 24) = *((_QWORD *)v13 + 3);
          *(_QWORD *)(v17 + 8) = v12;
          *(_QWORD *)(v12 + 32) = *((_QWORD *)v13 + 4);
          *((_QWORD *)v13 + 1) = 0;
          *((_QWORD *)v13 + 2) = v13;
          *((_QWORD *)v13 + 3) = v13;
          *((_QWORD *)v13 + 4) = 0;
        }
        v18 = *((_QWORD *)v13 + 5);
        v12 += 80;
        v13 += 20;
        *(_QWORD *)(v12 - 40) = v18;
        *(_QWORD *)(v12 - 32) = *((_QWORD *)v13 - 4);
        *(_QWORD *)(v12 - 24) = *((_QWORD *)v13 - 3);
        *(_QWORD *)(v12 - 16) = *((_QWORD *)v13 - 2);
        --v14;
      }
      while ( v14 );
      v19 = v101;
      if ( v101 <= 0 )
        v19 = 80;
      v20 = v99 + v19;
      if ( (int *)v103 != v8 )
      {
        if ( v99 == v20 )
          return;
        v21 = v20 - 80;
        v22 = v8 - 20;
        for ( i = v104 - 72; ; i -= 80 )
        {
          v24 = *(_QWORD *)(i + 8);
          v25 = i - 8;
          if ( *(_QWORD *)(v21 + 48) <= *((_QWORD *)v22 + 6) )
          {
            while ( v24 )
            {
              sub_261DCB0(*(_QWORD *)(v24 + 24));
              v71 = v24;
              v24 = *(_QWORD *)(v24 + 16);
              j_j___libc_free_0(v71);
            }
            *(_QWORD *)(i + 8) = 0;
            *(_QWORD *)(i + 16) = i;
            *(_QWORD *)(i + 24) = i;
            *(_QWORD *)(i + 32) = 0;
            if ( *(_QWORD *)(v21 + 16) )
            {
              *(_DWORD *)i = *(_DWORD *)(v21 + 8);
              v72 = *(_QWORD *)(v21 + 16);
              *(_QWORD *)(i + 8) = v72;
              *(_QWORD *)(i + 16) = *(_QWORD *)(v21 + 24);
              *(_QWORD *)(i + 24) = *(_QWORD *)(v21 + 32);
              *(_QWORD *)(v72 + 8) = i;
              *(_QWORD *)(i + 32) = *(_QWORD *)(v21 + 40);
              *(_QWORD *)(v21 + 16) = 0;
              *(_QWORD *)(v21 + 24) = v21 + 8;
              *(_QWORD *)(v21 + 32) = v21 + 8;
              *(_QWORD *)(v21 + 40) = 0;
            }
            *(_QWORD *)(i + 40) = *(_QWORD *)(v21 + 48);
            *(_QWORD *)(i + 48) = *(_QWORD *)(v21 + 56);
            *(_QWORD *)(i + 56) = *(_QWORD *)(v21 + 64);
            *(_QWORD *)(i + 64) = *(_QWORD *)(v21 + 72);
            if ( v99 == v21 )
              return;
            v21 -= 80;
          }
          else
          {
            while ( v24 )
            {
              sub_261DCB0(*(_QWORD *)(v24 + 24));
              v26 = v24;
              v24 = *(_QWORD *)(v24 + 16);
              j_j___libc_free_0(v26);
            }
            *(_QWORD *)(i + 8) = 0;
            *(_QWORD *)(i + 16) = i;
            *(_QWORD *)(i + 24) = i;
            *(_QWORD *)(i + 32) = 0;
            if ( *((_QWORD *)v22 + 2) )
            {
              *(_DWORD *)i = v22[2];
              v27 = *((_QWORD *)v22 + 2);
              *(_QWORD *)(i + 8) = v27;
              *(_QWORD *)(i + 16) = *((_QWORD *)v22 + 3);
              *(_QWORD *)(i + 24) = *((_QWORD *)v22 + 4);
              *(_QWORD *)(v27 + 8) = i;
              *(_QWORD *)(i + 32) = *((_QWORD *)v22 + 5);
              *((_QWORD *)v22 + 2) = 0;
              *((_QWORD *)v22 + 3) = v22 + 2;
              *((_QWORD *)v22 + 4) = v22 + 2;
              *((_QWORD *)v22 + 5) = 0;
            }
            *(_QWORD *)(i + 40) = *((_QWORD *)v22 + 6);
            *(_QWORD *)(i + 48) = *((_QWORD *)v22 + 7);
            *(_QWORD *)(i + 56) = *((_QWORD *)v22 + 8);
            *(_QWORD *)(i + 64) = *((_QWORD *)v22 + 9);
            if ( v22 == (int *)v103 )
            {
              v73 = 0xCCCCCCCCCCCCCCCDLL * ((v21 + 80 - v99) >> 4);
              if ( v21 + 80 - v99 > 0 )
              {
                v74 = v25;
                v75 = v21 + 8;
                v76 = v74 - 72;
                do
                {
                  v77 = *(_QWORD *)(v76 + 8);
                  while ( v77 )
                  {
                    sub_261DCB0(*(_QWORD *)(v77 + 24));
                    v78 = v77;
                    v77 = *(_QWORD *)(v77 + 16);
                    j_j___libc_free_0(v78);
                  }
                  *(_QWORD *)(v76 + 8) = 0;
                  *(_QWORD *)(v76 + 16) = v76;
                  *(_QWORD *)(v76 + 24) = v76;
                  *(_QWORD *)(v76 + 32) = 0;
                  if ( *(_QWORD *)(v75 + 8) )
                  {
                    *(_DWORD *)v76 = *(_DWORD *)v75;
                    v79 = *(_QWORD *)(v75 + 8);
                    *(_QWORD *)(v76 + 8) = v79;
                    *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
                    *(_QWORD *)(v76 + 24) = *(_QWORD *)(v75 + 24);
                    *(_QWORD *)(v79 + 8) = v76;
                    *(_QWORD *)(v76 + 32) = *(_QWORD *)(v75 + 32);
                    *(_QWORD *)(v75 + 8) = 0;
                    *(_QWORD *)(v75 + 16) = v75;
                    *(_QWORD *)(v75 + 24) = v75;
                    *(_QWORD *)(v75 + 32) = 0;
                  }
                  v80 = *(_QWORD *)(v75 + 40);
                  v76 -= 80;
                  v75 -= 80;
                  *(_QWORD *)(v76 + 120) = v80;
                  *(_QWORD *)(v76 + 128) = *(_QWORD *)(v75 + 128);
                  *(_QWORD *)(v76 + 136) = *(_QWORD *)(v75 + 136);
                  *(_QWORD *)(v76 + 144) = *(_QWORD *)(v75 + 144);
                  --v73;
                }
                while ( v73 );
              }
              return;
            }
            v22 -= 20;
          }
        }
      }
      v81 = 0xCCCCCCCCCCCCCCCDLL * (v19 >> 4);
      if ( v19 > 0 )
      {
        v82 = v20 - 72;
        v83 = v104 - 72;
        do
        {
          v84 = *(_QWORD *)(v83 + 8);
          while ( v84 )
          {
            sub_261DCB0(*(_QWORD *)(v84 + 24));
            v85 = v84;
            v84 = *(_QWORD *)(v84 + 16);
            j_j___libc_free_0(v85);
          }
          *(_QWORD *)(v83 + 8) = 0;
          *(_QWORD *)(v83 + 16) = v83;
          *(_QWORD *)(v83 + 24) = v83;
          *(_QWORD *)(v83 + 32) = 0;
          if ( *(_QWORD *)(v82 + 8) )
          {
            *(_DWORD *)v83 = *(_DWORD *)v82;
            v86 = *(_QWORD *)(v82 + 8);
            *(_QWORD *)(v83 + 8) = v86;
            *(_QWORD *)(v83 + 16) = *(_QWORD *)(v82 + 16);
            *(_QWORD *)(v83 + 24) = *(_QWORD *)(v82 + 24);
            *(_QWORD *)(v86 + 8) = v83;
            *(_QWORD *)(v83 + 32) = *(_QWORD *)(v82 + 32);
            *(_QWORD *)(v82 + 8) = 0;
            *(_QWORD *)(v82 + 16) = v82;
            *(_QWORD *)(v82 + 24) = v82;
            *(_QWORD *)(v82 + 32) = 0;
          }
          v87 = *(_QWORD *)(v82 + 40);
          v83 -= 80;
          v82 -= 80;
          *(_QWORD *)(v83 + 120) = v87;
          *(_QWORD *)(v83 + 128) = *(_QWORD *)(v82 + 128);
          *(_QWORD *)(v83 + 136) = *(_QWORD *)(v82 + 136);
          *(_QWORD *)(v83 + 144) = *(_QWORD *)(v82 + 144);
          --v81;
        }
        while ( v81 );
      }
      return;
    }
    if ( a5 >= a4 )
    {
      v95 = a5 / 2;
      v94 = &a2[20 * (a5 / 2)];
      v98 = sub_261ABA0(a1, (__int64)a2, (__int64)v94);
      v96 = 0xCCCCCCCCCCCCCCCDLL * ((v98 - a1) >> 4);
    }
    else
    {
      v96 = a4 / 2;
      v98 = a1 + 80 * (a4 / 2);
      v94 = (int *)sub_261AB30((__int64)a2, a3, v98);
      v95 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v94 - (char *)a2) >> 4);
    }
    v93 = v10 - v96;
    if ( v10 - v96 <= v95 || a7 < v95 )
    {
      if ( a7 < v93 )
      {
        v11 = (__int64)sub_261FC50(v98, a2, v94);
      }
      else
      {
        v11 = (__int64)v94;
        if ( v93 )
        {
          v90 = sub_261DE80(v98, (__int64)a2, v9);
          sub_261DE80((__int64)a2, (__int64)v94, v98);
          v55 = v90;
          v91 = v90 - v9;
          if ( v91 > 0 )
          {
            v88 = v9;
            v56 = v94 - 18;
            v57 = v55 - 72;
            v58 = 0xCCCCCCCCCCCCCCCDLL * (v91 >> 4);
            do
            {
              v59 = *((_QWORD *)v56 + 1);
              while ( v59 )
              {
                sub_261DCB0(*(_QWORD *)(v59 + 24));
                v60 = v59;
                v59 = *(_QWORD *)(v59 + 16);
                j_j___libc_free_0(v60);
              }
              *((_QWORD *)v56 + 1) = 0;
              *((_QWORD *)v56 + 2) = v56;
              *((_QWORD *)v56 + 3) = v56;
              *((_QWORD *)v56 + 4) = 0;
              if ( *(_QWORD *)(v57 + 8) )
              {
                *v56 = *(_DWORD *)v57;
                v61 = *(_QWORD *)(v57 + 8);
                *((_QWORD *)v56 + 1) = v61;
                *((_QWORD *)v56 + 2) = *(_QWORD *)(v57 + 16);
                *((_QWORD *)v56 + 3) = *(_QWORD *)(v57 + 24);
                *(_QWORD *)(v61 + 8) = v56;
                *((_QWORD *)v56 + 4) = *(_QWORD *)(v57 + 32);
                *(_QWORD *)(v57 + 8) = 0;
                *(_QWORD *)(v57 + 16) = v57;
                *(_QWORD *)(v57 + 24) = v57;
                *(_QWORD *)(v57 + 32) = 0;
              }
              v62 = *(_QWORD *)(v57 + 40);
              v56 -= 20;
              v57 -= 80;
              *((_QWORD *)v56 + 15) = v62;
              *((_QWORD *)v56 + 16) = *(_QWORD *)(v57 + 128);
              *((_QWORD *)v56 + 17) = *(_QWORD *)(v57 + 136);
              *((_QWORD *)v56 + 18) = *(_QWORD *)(v57 + 144);
              --v58;
            }
            while ( v58 );
            v9 = v88;
            v11 = (__int64)&v94[-4 * (v91 >> 4)];
          }
        }
      }
    }
    else
    {
      v11 = v98;
      if ( v95 )
      {
        v63 = sub_261DE80((__int64)a2, (__int64)v94, v9);
        if ( (__int64)a2 - v98 > 0 )
        {
          v92 = v63;
          v64 = a2 - 18;
          v65 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)a2 - v98) >> 4);
          v89 = v9;
          v66 = v94 - 18;
          do
          {
            v67 = *((_QWORD *)v66 + 1);
            while ( v67 )
            {
              sub_261DCB0(*(_QWORD *)(v67 + 24));
              v68 = v67;
              v67 = *(_QWORD *)(v67 + 16);
              j_j___libc_free_0(v68);
            }
            *((_QWORD *)v66 + 1) = 0;
            *((_QWORD *)v66 + 2) = v66;
            *((_QWORD *)v66 + 3) = v66;
            *((_QWORD *)v66 + 4) = 0;
            if ( *((_QWORD *)v64 + 1) )
            {
              *v66 = *v64;
              v69 = *((_QWORD *)v64 + 1);
              *((_QWORD *)v66 + 1) = v69;
              *((_QWORD *)v66 + 2) = *((_QWORD *)v64 + 2);
              *((_QWORD *)v66 + 3) = *((_QWORD *)v64 + 3);
              *(_QWORD *)(v69 + 8) = v66;
              *((_QWORD *)v66 + 4) = *((_QWORD *)v64 + 4);
              *((_QWORD *)v64 + 1) = 0;
              *((_QWORD *)v64 + 2) = v64;
              *((_QWORD *)v64 + 3) = v64;
              *((_QWORD *)v64 + 4) = 0;
            }
            v70 = *((_QWORD *)v64 + 5);
            v66 -= 20;
            v64 -= 20;
            *((_QWORD *)v66 + 15) = v70;
            *((_QWORD *)v66 + 16) = *((_QWORD *)v64 + 16);
            *((_QWORD *)v66 + 17) = *((_QWORD *)v64 + 17);
            *((_QWORD *)v66 + 18) = *((_QWORD *)v64 + 18);
            --v65;
          }
          while ( v65 );
          v63 = v92;
          v9 = v89;
        }
        v11 = sub_261DE80(v9, v63, v98);
      }
    }
    sub_2623DA0(v103, v98, v11, v96, v95, v9, a7);
    a4 = v93;
    a2 = v94;
    a6 = v9;
    a1 = v11;
    a3 = v104;
    a5 = v100 - v95;
  }
  v36 = (__int64)a2 - a1;
  v37 = a1 + 8;
  v38 = a6 + 8;
  if ( v36 > 0 )
  {
    v102 = a6;
    v39 = a1 + 8;
    v40 = 0xCCCCCCCCCCCCCCCDLL * (v36 >> 4);
    do
    {
      v41 = *(_QWORD *)(v38 + 8);
      while ( v41 )
      {
        sub_261DCB0(*(_QWORD *)(v41 + 24));
        v42 = v41;
        v41 = *(_QWORD *)(v41 + 16);
        j_j___libc_free_0(v42);
      }
      *(_QWORD *)(v38 + 8) = 0;
      *(_QWORD *)(v38 + 16) = v38;
      *(_QWORD *)(v38 + 24) = v38;
      *(_QWORD *)(v38 + 32) = 0;
      if ( *(_QWORD *)(v39 + 8) )
      {
        *(_DWORD *)v38 = *(_DWORD *)v39;
        v43 = *(_QWORD *)(v39 + 8);
        *(_QWORD *)(v38 + 8) = v43;
        *(_QWORD *)(v38 + 16) = *(_QWORD *)(v39 + 16);
        *(_QWORD *)(v38 + 24) = *(_QWORD *)(v39 + 24);
        *(_QWORD *)(v43 + 8) = v38;
        *(_QWORD *)(v38 + 32) = *(_QWORD *)(v39 + 32);
        *(_QWORD *)(v39 + 8) = 0;
        *(_QWORD *)(v39 + 16) = v39;
        *(_QWORD *)(v39 + 24) = v39;
        *(_QWORD *)(v39 + 32) = 0;
      }
      v44 = *(_QWORD *)(v39 + 40);
      v38 += 80;
      v39 += 80;
      *(_QWORD *)(v38 - 40) = v44;
      *(_QWORD *)(v38 - 32) = *(_QWORD *)(v39 - 32);
      *(_QWORD *)(v38 - 24) = *(_QWORD *)(v39 - 24);
      *(_QWORD *)(v38 - 16) = *(_QWORD *)(v39 - 16);
      --v40;
    }
    while ( v40 );
    v45 = v102;
    v46 = v102 + v36;
    if ( v102 != v102 + v36 )
    {
      v47 = v97;
      while ( (int *)v104 != v47 )
      {
        v48 = *(_QWORD *)(v37 + 8);
        if ( *((_QWORD *)v47 + 6) <= *(_QWORD *)(v45 + 48) )
        {
          while ( v48 )
          {
            sub_261DCB0(*(_QWORD *)(v48 + 24));
            v52 = v48;
            v48 = *(_QWORD *)(v48 + 16);
            j_j___libc_free_0(v52);
          }
          *(_QWORD *)(v37 + 8) = 0;
          *(_QWORD *)(v37 + 16) = v37;
          *(_QWORD *)(v37 + 24) = v37;
          *(_QWORD *)(v37 + 32) = 0;
          if ( *(_QWORD *)(v45 + 16) )
          {
            *(_DWORD *)v37 = *(_DWORD *)(v45 + 8);
            v53 = *(_QWORD *)(v45 + 16);
            *(_QWORD *)(v37 + 8) = v53;
            *(_QWORD *)(v37 + 16) = *(_QWORD *)(v45 + 24);
            *(_QWORD *)(v37 + 24) = *(_QWORD *)(v45 + 32);
            *(_QWORD *)(v53 + 8) = v37;
            *(_QWORD *)(v37 + 32) = *(_QWORD *)(v45 + 40);
            *(_QWORD *)(v45 + 16) = 0;
            *(_QWORD *)(v45 + 24) = v45 + 8;
            *(_QWORD *)(v45 + 32) = v45 + 8;
            *(_QWORD *)(v45 + 40) = 0;
          }
          v54 = *(_QWORD *)(v45 + 48);
          v45 += 80;
          *(_QWORD *)(v37 + 40) = v54;
          *(_QWORD *)(v37 + 48) = *(_QWORD *)(v45 - 24);
          *(_QWORD *)(v37 + 56) = *(_QWORD *)(v45 - 16);
          *(_QWORD *)(v37 + 64) = *(_QWORD *)(v45 - 8);
        }
        else
        {
          while ( v48 )
          {
            sub_261DCB0(*(_QWORD *)(v48 + 24));
            v49 = v48;
            v48 = *(_QWORD *)(v48 + 16);
            j_j___libc_free_0(v49);
          }
          *(_QWORD *)(v37 + 8) = 0;
          *(_QWORD *)(v37 + 16) = v37;
          *(_QWORD *)(v37 + 24) = v37;
          *(_QWORD *)(v37 + 32) = 0;
          if ( *((_QWORD *)v47 + 2) )
          {
            *(_DWORD *)v37 = v47[2];
            v50 = *((_QWORD *)v47 + 2);
            *(_QWORD *)(v37 + 8) = v50;
            *(_QWORD *)(v37 + 16) = *((_QWORD *)v47 + 3);
            *(_QWORD *)(v37 + 24) = *((_QWORD *)v47 + 4);
            *(_QWORD *)(v50 + 8) = v37;
            *(_QWORD *)(v37 + 32) = *((_QWORD *)v47 + 5);
            *((_QWORD *)v47 + 2) = 0;
            *((_QWORD *)v47 + 3) = v47 + 2;
            *((_QWORD *)v47 + 4) = v47 + 2;
            *((_QWORD *)v47 + 5) = 0;
          }
          v51 = *((_QWORD *)v47 + 6);
          v47 += 20;
          *(_QWORD *)(v37 + 40) = v51;
          *(_QWORD *)(v37 + 48) = *((_QWORD *)v47 - 3);
          *(_QWORD *)(v37 + 56) = *((_QWORD *)v47 - 2);
          *(_QWORD *)(v37 + 64) = *((_QWORD *)v47 - 1);
        }
        v37 += 80;
        if ( v45 == v46 )
          return;
      }
      v28 = v46 - v45;
      v29 = v37;
      v30 = v45 + 8;
      v31 = 0xCCCCCCCCCCCCCCCDLL * (v28 >> 4);
      if ( v28 > 0 )
      {
        do
        {
          v32 = *(_QWORD *)(v29 + 8);
          while ( v32 )
          {
            sub_261DCB0(*(_QWORD *)(v32 + 24));
            v33 = v32;
            v32 = *(_QWORD *)(v32 + 16);
            j_j___libc_free_0(v33);
          }
          *(_QWORD *)(v29 + 8) = 0;
          *(_QWORD *)(v29 + 16) = v29;
          *(_QWORD *)(v29 + 24) = v29;
          *(_QWORD *)(v29 + 32) = 0;
          if ( *(_QWORD *)(v30 + 8) )
          {
            *(_DWORD *)v29 = *(_DWORD *)v30;
            v34 = *(_QWORD *)(v30 + 8);
            *(_QWORD *)(v29 + 8) = v34;
            *(_QWORD *)(v29 + 16) = *(_QWORD *)(v30 + 16);
            *(_QWORD *)(v29 + 24) = *(_QWORD *)(v30 + 24);
            *(_QWORD *)(v34 + 8) = v29;
            *(_QWORD *)(v29 + 32) = *(_QWORD *)(v30 + 32);
            *(_QWORD *)(v30 + 8) = 0;
            *(_QWORD *)(v30 + 16) = v30;
            *(_QWORD *)(v30 + 24) = v30;
            *(_QWORD *)(v30 + 32) = 0;
          }
          v35 = *(_QWORD *)(v30 + 40);
          v29 += 80;
          v30 += 80;
          *(_QWORD *)(v29 - 40) = v35;
          *(_QWORD *)(v29 - 32) = *(_QWORD *)(v30 - 32);
          *(_QWORD *)(v29 - 24) = *(_QWORD *)(v30 - 24);
          *(_QWORD *)(v29 - 16) = *(_QWORD *)(v30 - 16);
          --v31;
        }
        while ( v31 );
      }
    }
  }
}

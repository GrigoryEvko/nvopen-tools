// Function: sub_2429000
// Address: 0x2429000
//
void __fastcall sub_2429000(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int64 *a6,
        __int64 a7)
{
  unsigned __int64 *v7; // r12
  unsigned __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // r10
  __int64 v13; // r11
  __int64 v14; // r8
  __int64 v15; // r14
  __int64 v16; // r13
  __int64 v17; // r12
  unsigned __int64 *v18; // r15
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // r15
  unsigned __int64 *v23; // rax
  __int64 v24; // r14
  unsigned __int64 *v25; // r15
  unsigned __int64 *i; // r13
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rdi
  __int64 v31; // r15
  unsigned __int64 *v32; // r14
  __int64 v33; // r12
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rdi
  unsigned __int64 *v36; // rbx
  __int64 v37; // r13
  unsigned __int64 *v38; // r15
  unsigned __int64 *v39; // r14
  __int64 v40; // rbx
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rdi
  __int64 v43; // rsi
  unsigned __int64 *v44; // r15
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rdx
  unsigned __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rbx
  unsigned __int64 *v51; // r15
  unsigned __int64 *v52; // r13
  unsigned __int64 v53; // rax
  unsigned __int64 v54; // rdi
  __int64 v55; // rbx
  unsigned __int64 *v56; // r15
  __int64 v57; // r13
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rdi
  __int64 v60; // [rsp+10h] [rbp-60h]
  __int64 v61; // [rsp+18h] [rbp-58h]
  __int64 v62; // [rsp+20h] [rbp-50h]
  __int64 v63; // [rsp+20h] [rbp-50h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v65; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v66; // [rsp+28h] [rbp-48h]
  __int64 v67; // [rsp+30h] [rbp-40h]
  __int64 v68; // [rsp+30h] [rbp-40h]
  __int64 v69; // [rsp+30h] [rbp-40h]
  __int64 v70; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (unsigned __int64 *)a1;
    v8 = a6;
    v70 = a3;
    v9 = a7;
    if ( a5 <= a7 )
      v9 = a5;
    if ( v9 >= a4 )
      break;
    v10 = a5;
    if ( a5 <= a7 )
    {
      v16 = a3 - a2;
      if ( a3 - a2 <= 0 )
        return;
      v68 = a1;
      v17 = (a3 - a2) >> 3;
      v18 = (unsigned __int64 *)a2;
      v65 = a6;
      do
      {
        v19 = *v18;
        *v18 = 0;
        v20 = *v8;
        *v8 = v19;
        if ( v20 )
          j_j___libc_free_0(v20);
        ++v18;
        ++v8;
        --v17;
      }
      while ( v17 );
      v21 = 8;
      if ( v16 > 0 )
        v21 = v16;
      v22 = v21;
      v23 = (unsigned __int64 *)((char *)v65 + v21);
      if ( v68 != a2 )
      {
        if ( v65 == v23 )
          return;
        v24 = a2 - 8;
        v25 = v23 - 1;
        for ( i = (unsigned __int64 *)(v70 - 8); ; --i )
        {
          v28 = *(_QWORD *)v24;
          v29 = *v25;
          if ( *(_QWORD *)(*v25 + 16) > *(_QWORD *)(*(_QWORD *)v24 + 16LL) )
          {
            *(_QWORD *)v24 = 0;
            v27 = *i;
            *i = v28;
            if ( v27 )
              j_j___libc_free_0(v27);
            if ( v68 == v24 )
            {
              v50 = v25 + 1 - v65;
              if ( (char *)(v25 + 1) - (char *)v65 > 0 )
              {
                v51 = &v25[-v50];
                v52 = &i[-v50];
                do
                {
                  v53 = v51[v50];
                  v51[v50] = 0;
                  v54 = v52[v50 - 1];
                  v52[v50 - 1] = v53;
                  if ( v54 )
                    j_j___libc_free_0(v54);
                  --v50;
                }
                while ( v50 );
              }
              return;
            }
            v24 -= 8;
          }
          else
          {
            *v25 = 0;
            v30 = *i;
            *i = v29;
            if ( v30 )
              j_j___libc_free_0(v30);
            if ( v65 == v25 )
              return;
            --v25;
          }
        }
      }
      v55 = v22 >> 3;
      v56 = &v23[-(v22 >> 3)];
      v57 = -8 * v55 + v70;
      do
      {
        v58 = v56[v55 - 1];
        v56[v55 - 1] = 0;
        v59 = *(_QWORD *)(v57 + 8 * v55 - 8);
        *(_QWORD *)(v57 + 8 * v55 - 8) = v58;
        if ( v59 )
          j_j___libc_free_0(v59);
        --v55;
      }
      while ( v55 );
      return;
    }
    if ( a5 >= a4 )
    {
      v63 = a5 / 2;
      v64 = a2 + 8 * (a5 / 2);
      v49 = sub_24259B0(a1, a2, v64);
      v14 = v63;
      v11 = v49;
      v67 = (v49 - a1) >> 3;
    }
    else
    {
      v67 = a4 / 2;
      v11 = a1 + 8 * (a4 / 2);
      v64 = sub_2425960(a2, a3, v11);
      v14 = (v64 - a2) >> 3;
    }
    v60 = v12 - v67;
    v61 = v13;
    v62 = v14;
    v15 = sub_2428400(v11, a2, v64, v12 - v67, v14, v8, v13);
    sub_2429000(a1, v11, v15, v67, v62, (_DWORD)v8, v61);
    a6 = v8;
    a2 = v64;
    a1 = v15;
    a7 = v61;
    a3 = v70;
    a5 = v10 - v62;
    a4 = v60;
  }
  v37 = a2;
  v38 = (unsigned __int64 *)a1;
  v39 = a6;
  v69 = a2 - a1;
  if ( a2 - a1 > 0 )
  {
    v66 = a6;
    v40 = (a2 - a1) >> 3;
    do
    {
      v41 = *v38;
      *v38 = 0;
      v42 = *v39;
      *v39 = v41;
      if ( v42 )
        j_j___libc_free_0(v42);
      ++v38;
      ++v39;
      --v40;
    }
    while ( v40 );
    v43 = v69;
    v36 = v66;
    if ( v69 <= 0 )
      v43 = 8;
    v44 = (unsigned __int64 *)((char *)v66 + v43);
    if ( v66 != (unsigned __int64 *)((char *)v66 + v43) )
    {
      while ( v70 != v37 )
      {
        v46 = *v36;
        v47 = *(_QWORD *)v37;
        if ( *(_QWORD *)(*(_QWORD *)v37 + 16LL) > *(_QWORD *)(*v36 + 16) )
        {
          *(_QWORD *)v37 = 0;
          v45 = *v7;
          *v7 = v47;
          if ( v45 )
            j_j___libc_free_0(v45);
          v37 += 8;
        }
        else
        {
          *v36 = 0;
          v48 = *v7;
          *v7 = v46;
          if ( v48 )
            j_j___libc_free_0(v48);
          ++v36;
        }
        ++v7;
        if ( v36 == v44 )
          return;
      }
      v31 = (char *)v44 - (char *)v36;
      v32 = v7;
      v33 = v31 >> 3;
      if ( v31 > 0 )
      {
        do
        {
          v34 = *v36;
          *v36 = 0;
          v35 = *v32;
          *v32 = v34;
          if ( v35 )
            j_j___libc_free_0(v35);
          ++v36;
          ++v32;
          --v33;
        }
        while ( v33 );
      }
    }
  }
}

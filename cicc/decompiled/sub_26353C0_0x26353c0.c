// Function: sub_26353C0
// Address: 0x26353c0
//
void __fastcall sub_26353C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r10
  __int64 v10; // rbx
  __int64 v11; // rbx
  __int64 v12; // r13
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // r12
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 i; // r12
  unsigned __int64 v23; // rbx
  __int64 v24; // r15
  unsigned __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // r12
  __int64 v28; // r13
  unsigned __int64 v29; // r15
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r12
  __int64 v35; // r15
  __int64 v36; // r13
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // rdi
  __int64 v39; // rdx
  unsigned __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rbx
  __int64 v44; // r14
  unsigned __int64 v45; // r13
  unsigned __int64 v46; // r12
  unsigned __int64 v47; // rdi
  __int64 v48; // rdx
  unsigned __int64 v49; // r12
  __int64 v50; // r14
  __int64 v51; // rbx
  unsigned __int64 v52; // r13
  unsigned __int64 v53; // rdi
  __int64 v54; // rdx
  unsigned __int64 v55; // rdi
  __int64 v56; // rcx
  unsigned __int64 v57; // r12
  __int64 v58; // r15
  __int64 v59; // r13
  unsigned __int64 v60; // rbx
  unsigned __int64 v61; // rdi
  __int64 v62; // rax
  unsigned __int64 v63; // r14
  __int64 v64; // r13
  __int64 v65; // r12
  unsigned __int64 v66; // rbx
  unsigned __int64 v67; // rdi
  __int64 v68; // rax
  __int64 v69; // [rsp+8h] [rbp-88h]
  __int64 v70; // [rsp+10h] [rbp-80h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+18h] [rbp-78h]
  __int64 v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+28h] [rbp-68h]
  __int64 v76; // [rsp+30h] [rbp-60h]
  __int64 v77; // [rsp+38h] [rbp-58h]
  __int64 v78; // [rsp+40h] [rbp-50h]
  __int64 v79; // [rsp+40h] [rbp-50h]
  __int64 v80; // [rsp+48h] [rbp-48h]
  __int64 v81; // [rsp+48h] [rbp-48h]
  __int64 v82; // [rsp+48h] [rbp-48h]
  __int64 v83; // [rsp+50h] [rbp-40h]
  __int64 v84; // [rsp+50h] [rbp-40h]
  __int64 v85; // [rsp+58h] [rbp-38h]
  __int64 v86; // [rsp+58h] [rbp-38h]

  while ( 1 )
  {
    v7 = a5;
    v8 = a6;
    v83 = a1;
    v85 = a3;
    v80 = a5;
    if ( a5 > a7 )
      v7 = a7;
    if ( v7 >= a4 )
    {
      v84 = a2 - a1;
      v27 = a1 + 8;
      v28 = a6 + 8;
      if ( a2 - a1 > 0 )
      {
        v82 = a1 + 8;
        v29 = 0xAAAAAAAAAAAAAAABLL * ((a2 - a1) >> 4);
        do
        {
          v30 = *(_QWORD *)(v28 + 8);
          while ( v30 )
          {
            sub_261DCB0(*(_QWORD *)(v30 + 24));
            v31 = v30;
            v30 = *(_QWORD *)(v30 + 16);
            j_j___libc_free_0(v31);
          }
          *(_QWORD *)(v28 + 8) = 0;
          *(_QWORD *)(v28 + 16) = v28;
          *(_QWORD *)(v28 + 24) = v28;
          *(_QWORD *)(v28 + 32) = 0;
          if ( *(_QWORD *)(v27 + 8) )
          {
            *(_DWORD *)v28 = *(_DWORD *)v27;
            v32 = *(_QWORD *)(v27 + 8);
            *(_QWORD *)(v28 + 8) = v32;
            *(_QWORD *)(v28 + 16) = *(_QWORD *)(v27 + 16);
            *(_QWORD *)(v28 + 24) = *(_QWORD *)(v27 + 24);
            *(_QWORD *)(v32 + 8) = v28;
            *(_QWORD *)(v28 + 32) = *(_QWORD *)(v27 + 32);
            *(_QWORD *)(v27 + 8) = 0;
            *(_QWORD *)(v27 + 16) = v27;
            *(_QWORD *)(v27 + 24) = v27;
            *(_QWORD *)(v27 + 32) = 0;
          }
          v28 += 48;
          v27 += 48;
          --v29;
        }
        while ( v29 );
        v33 = v84;
        v34 = v82;
        if ( v84 <= 0 )
          v33 = 48;
        v35 = v8 + v33;
        if ( v8 != v8 + v33 )
        {
          v36 = a2;
          do
          {
            if ( v85 == v36 )
            {
              sub_1888CD0(v8, v35, v34 - 8);
              return;
            }
            v37 = *(_QWORD *)(v34 + 8);
            if ( *(_QWORD *)(v36 + 40) >= *(_QWORD *)(v8 + 40) )
            {
              while ( v37 )
              {
                sub_261DCB0(*(_QWORD *)(v37 + 24));
                v40 = v37;
                v37 = *(_QWORD *)(v37 + 16);
                j_j___libc_free_0(v40);
              }
              *(_QWORD *)(v34 + 8) = 0;
              *(_QWORD *)(v34 + 16) = v34;
              *(_QWORD *)(v34 + 24) = v34;
              *(_QWORD *)(v34 + 32) = 0;
              if ( *(_QWORD *)(v8 + 16) )
              {
                *(_DWORD *)v34 = *(_DWORD *)(v8 + 8);
                v41 = *(_QWORD *)(v8 + 16);
                *(_QWORD *)(v34 + 8) = v41;
                *(_QWORD *)(v34 + 16) = *(_QWORD *)(v8 + 24);
                *(_QWORD *)(v34 + 24) = *(_QWORD *)(v8 + 32);
                *(_QWORD *)(v41 + 8) = v34;
                *(_QWORD *)(v34 + 32) = *(_QWORD *)(v8 + 40);
                *(_QWORD *)(v8 + 16) = 0;
                *(_QWORD *)(v8 + 24) = v8 + 8;
                *(_QWORD *)(v8 + 32) = v8 + 8;
                *(_QWORD *)(v8 + 40) = 0;
              }
              v8 += 48;
            }
            else
            {
              while ( v37 )
              {
                sub_261DCB0(*(_QWORD *)(v37 + 24));
                v38 = v37;
                v37 = *(_QWORD *)(v37 + 16);
                j_j___libc_free_0(v38);
              }
              *(_QWORD *)(v34 + 8) = 0;
              *(_QWORD *)(v34 + 16) = v34;
              *(_QWORD *)(v34 + 24) = v34;
              *(_QWORD *)(v34 + 32) = 0;
              if ( *(_QWORD *)(v36 + 16) )
              {
                *(_DWORD *)v34 = *(_DWORD *)(v36 + 8);
                v39 = *(_QWORD *)(v36 + 16);
                *(_QWORD *)(v34 + 8) = v39;
                *(_QWORD *)(v34 + 16) = *(_QWORD *)(v36 + 24);
                *(_QWORD *)(v34 + 24) = *(_QWORD *)(v36 + 32);
                *(_QWORD *)(v39 + 8) = v34;
                *(_QWORD *)(v34 + 32) = *(_QWORD *)(v36 + 40);
                *(_QWORD *)(v36 + 16) = 0;
                *(_QWORD *)(v36 + 24) = v36 + 8;
                *(_QWORD *)(v36 + 32) = v36 + 8;
                *(_QWORD *)(v36 + 40) = 0;
              }
              v36 += 48;
            }
            v34 += 48;
          }
          while ( v8 != v35 );
        }
      }
      return;
    }
    if ( a5 <= a7 )
      break;
    if ( a5 >= a4 )
    {
      v76 = a5 / 2;
      v75 = a2 + 16 * (a5 / 2 + ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
      v78 = sub_261AC80(a1, a2, v75);
      v77 = 0xAAAAAAAAAAAAAAABLL * ((v78 - a1) >> 4);
    }
    else
    {
      v77 = a4 / 2;
      v78 = a1 + 16 * (a4 / 2 + ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
      v75 = sub_261AC10(a2, a3, v78);
      v76 = 0xAAAAAAAAAAAAAAABLL * ((v75 - a2) >> 4);
    }
    v74 = v9 - v77;
    if ( v9 - v77 <= v76 || a7 < v76 )
    {
      if ( a7 < v74 )
      {
        v10 = sub_2634FC0(v78, a2, v75);
      }
      else
      {
        v10 = v75;
        if ( v74 )
        {
          v71 = sub_1888CD0(v78, a2, v8);
          sub_1888CD0(a2, v75, v78);
          v42 = v71;
          v72 = v71 - v8;
          if ( v72 > 0 )
          {
            v69 = v8;
            v43 = v75 - 40;
            v44 = v42 - 40;
            v45 = 0xAAAAAAAAAAAAAAABLL * (v72 >> 4);
            do
            {
              v46 = *(_QWORD *)(v43 + 8);
              while ( v46 )
              {
                sub_261DCB0(*(_QWORD *)(v46 + 24));
                v47 = v46;
                v46 = *(_QWORD *)(v46 + 16);
                j_j___libc_free_0(v47);
              }
              *(_QWORD *)(v43 + 8) = 0;
              *(_QWORD *)(v43 + 16) = v43;
              *(_QWORD *)(v43 + 24) = v43;
              *(_QWORD *)(v43 + 32) = 0;
              if ( *(_QWORD *)(v44 + 8) )
              {
                *(_DWORD *)v43 = *(_DWORD *)v44;
                v48 = *(_QWORD *)(v44 + 8);
                *(_QWORD *)(v43 + 8) = v48;
                *(_QWORD *)(v43 + 16) = *(_QWORD *)(v44 + 16);
                *(_QWORD *)(v43 + 24) = *(_QWORD *)(v44 + 24);
                *(_QWORD *)(v48 + 8) = v43;
                *(_QWORD *)(v43 + 32) = *(_QWORD *)(v44 + 32);
                *(_QWORD *)(v44 + 8) = 0;
                *(_QWORD *)(v44 + 16) = v44;
                *(_QWORD *)(v44 + 24) = v44;
                *(_QWORD *)(v44 + 32) = 0;
              }
              v43 -= 48;
              v44 -= 48;
              --v45;
            }
            while ( v45 );
            v8 = v69;
            v10 = v75 - 16 * (v72 >> 4);
          }
        }
      }
    }
    else
    {
      v10 = v78;
      if ( v76 )
      {
        v73 = sub_1888CD0(a2, v75, v8);
        v49 = 0xAAAAAAAAAAAAAAABLL * ((a2 - v78) >> 4);
        if ( a2 - v78 > 0 )
        {
          v70 = v8;
          v50 = a2 - 40;
          v51 = v75 - 40;
          do
          {
            v52 = *(_QWORD *)(v51 + 8);
            while ( v52 )
            {
              sub_261DCB0(*(_QWORD *)(v52 + 24));
              v53 = v52;
              v52 = *(_QWORD *)(v52 + 16);
              j_j___libc_free_0(v53);
            }
            *(_QWORD *)(v51 + 8) = 0;
            *(_QWORD *)(v51 + 16) = v51;
            *(_QWORD *)(v51 + 24) = v51;
            *(_QWORD *)(v51 + 32) = 0;
            if ( *(_QWORD *)(v50 + 8) )
            {
              *(_DWORD *)v51 = *(_DWORD *)v50;
              v54 = *(_QWORD *)(v50 + 8);
              *(_QWORD *)(v51 + 8) = v54;
              *(_QWORD *)(v51 + 16) = *(_QWORD *)(v50 + 16);
              *(_QWORD *)(v51 + 24) = *(_QWORD *)(v50 + 24);
              *(_QWORD *)(v54 + 8) = v51;
              *(_QWORD *)(v51 + 32) = *(_QWORD *)(v50 + 32);
              *(_QWORD *)(v50 + 8) = 0;
              *(_QWORD *)(v50 + 16) = v50;
              *(_QWORD *)(v50 + 24) = v50;
              *(_QWORD *)(v50 + 32) = 0;
            }
            v51 -= 48;
            v50 -= 48;
            --v49;
          }
          while ( v49 );
          v8 = v70;
        }
        v10 = sub_1888CD0(v8, v73, v78);
      }
    }
    sub_26353C0(v83, v78, v10, v77, v76, v8, a7);
    a4 = v74;
    a2 = v75;
    a6 = v8;
    a1 = v10;
    a3 = v85;
    a5 = v80 - v76;
  }
  v81 = a3 - a2;
  if ( a3 - a2 <= 0 )
    return;
  v79 = a2;
  v11 = a6 + 8;
  v12 = a2 + 8;
  v13 = 0xAAAAAAAAAAAAAAABLL * ((a3 - a2) >> 4);
  do
  {
    v14 = *(_QWORD *)(v11 + 8);
    while ( v14 )
    {
      sub_261DCB0(*(_QWORD *)(v14 + 24));
      v15 = v14;
      v14 = *(_QWORD *)(v14 + 16);
      j_j___libc_free_0(v15);
    }
    *(_QWORD *)(v11 + 8) = 0;
    *(_QWORD *)(v11 + 16) = v11;
    *(_QWORD *)(v11 + 24) = v11;
    *(_QWORD *)(v11 + 32) = 0;
    if ( *(_QWORD *)(v12 + 8) )
    {
      *(_DWORD *)v11 = *(_DWORD *)v12;
      v16 = *(_QWORD *)(v12 + 8);
      *(_QWORD *)(v11 + 8) = v16;
      *(_QWORD *)(v11 + 16) = *(_QWORD *)(v12 + 16);
      *(_QWORD *)(v11 + 24) = *(_QWORD *)(v12 + 24);
      *(_QWORD *)(v16 + 8) = v11;
      *(_QWORD *)(v11 + 32) = *(_QWORD *)(v12 + 32);
      *(_QWORD *)(v12 + 8) = 0;
      *(_QWORD *)(v12 + 16) = v12;
      *(_QWORD *)(v12 + 24) = v12;
      *(_QWORD *)(v12 + 32) = 0;
    }
    v11 += 48;
    v12 += 48;
    --v13;
  }
  while ( v13 );
  v17 = v81;
  if ( v81 <= 0 )
    v17 = 48;
  v18 = v8 + v17;
  if ( v83 != v79 )
  {
    if ( v8 == v18 )
      return;
    v19 = v85;
    v86 = v8;
    v20 = v18 - 48;
    v21 = v79 - 48;
    for ( i = v19 - 40; ; i -= 48 )
    {
      v23 = *(_QWORD *)(i + 8);
      v24 = i - 8;
      if ( *(_QWORD *)(v20 + 40) >= *(_QWORD *)(v21 + 40) )
      {
        while ( v23 )
        {
          sub_261DCB0(*(_QWORD *)(v23 + 24));
          v55 = v23;
          v23 = *(_QWORD *)(v23 + 16);
          j_j___libc_free_0(v55);
        }
        *(_QWORD *)(i + 8) = 0;
        *(_QWORD *)(i + 16) = i;
        *(_QWORD *)(i + 24) = i;
        *(_QWORD *)(i + 32) = 0;
        if ( *(_QWORD *)(v20 + 16) )
        {
          *(_DWORD *)i = *(_DWORD *)(v20 + 8);
          v56 = *(_QWORD *)(v20 + 16);
          *(_QWORD *)(i + 8) = v56;
          *(_QWORD *)(i + 16) = *(_QWORD *)(v20 + 24);
          *(_QWORD *)(i + 24) = *(_QWORD *)(v20 + 32);
          *(_QWORD *)(v56 + 8) = i;
          *(_QWORD *)(i + 32) = *(_QWORD *)(v20 + 40);
          *(_QWORD *)(v20 + 16) = 0;
          *(_QWORD *)(v20 + 24) = v20 + 8;
          *(_QWORD *)(v20 + 32) = v20 + 8;
          *(_QWORD *)(v20 + 40) = 0;
        }
        if ( v86 == v20 )
          return;
        v20 -= 48;
      }
      else
      {
        while ( v23 )
        {
          sub_261DCB0(*(_QWORD *)(v23 + 24));
          v25 = v23;
          v23 = *(_QWORD *)(v23 + 16);
          j_j___libc_free_0(v25);
        }
        *(_QWORD *)(i + 8) = 0;
        *(_QWORD *)(i + 16) = i;
        *(_QWORD *)(i + 24) = i;
        *(_QWORD *)(i + 32) = 0;
        if ( *(_QWORD *)(v21 + 16) )
        {
          *(_DWORD *)i = *(_DWORD *)(v21 + 8);
          v26 = *(_QWORD *)(v21 + 16);
          *(_QWORD *)(i + 8) = v26;
          *(_QWORD *)(i + 16) = *(_QWORD *)(v21 + 24);
          *(_QWORD *)(i + 24) = *(_QWORD *)(v21 + 32);
          *(_QWORD *)(v26 + 8) = i;
          *(_QWORD *)(i + 32) = *(_QWORD *)(v21 + 40);
          *(_QWORD *)(v21 + 16) = 0;
          *(_QWORD *)(v21 + 24) = v21 + 8;
          *(_QWORD *)(v21 + 32) = v21 + 8;
          *(_QWORD *)(v21 + 40) = 0;
        }
        if ( v21 == v83 )
        {
          v57 = 0xAAAAAAAAAAAAAAABLL * ((v20 + 48 - v86) >> 4);
          if ( v20 + 48 - v86 > 0 )
          {
            v58 = v24 - 40;
            v59 = v20 + 8;
            do
            {
              v60 = *(_QWORD *)(v58 + 8);
              while ( v60 )
              {
                sub_261DCB0(*(_QWORD *)(v60 + 24));
                v61 = v60;
                v60 = *(_QWORD *)(v60 + 16);
                j_j___libc_free_0(v61);
              }
              *(_QWORD *)(v58 + 8) = 0;
              *(_QWORD *)(v58 + 16) = v58;
              *(_QWORD *)(v58 + 24) = v58;
              *(_QWORD *)(v58 + 32) = 0;
              if ( *(_QWORD *)(v59 + 8) )
              {
                *(_DWORD *)v58 = *(_DWORD *)v59;
                v62 = *(_QWORD *)(v59 + 8);
                *(_QWORD *)(v58 + 8) = v62;
                *(_QWORD *)(v58 + 16) = *(_QWORD *)(v59 + 16);
                *(_QWORD *)(v58 + 24) = *(_QWORD *)(v59 + 24);
                *(_QWORD *)(v62 + 8) = v58;
                *(_QWORD *)(v58 + 32) = *(_QWORD *)(v59 + 32);
                *(_QWORD *)(v59 + 8) = 0;
                *(_QWORD *)(v59 + 16) = v59;
                *(_QWORD *)(v59 + 24) = v59;
                *(_QWORD *)(v59 + 32) = 0;
              }
              v58 -= 48;
              v59 -= 48;
              --v57;
            }
            while ( v57 );
          }
          return;
        }
        v21 -= 48;
      }
    }
  }
  v63 = 0xAAAAAAAAAAAAAAABLL * (v17 >> 4);
  if ( v17 > 0 )
  {
    v64 = v18 - 40;
    v65 = v85 - 40;
    do
    {
      v66 = *(_QWORD *)(v65 + 8);
      while ( v66 )
      {
        sub_261DCB0(*(_QWORD *)(v66 + 24));
        v67 = v66;
        v66 = *(_QWORD *)(v66 + 16);
        j_j___libc_free_0(v67);
      }
      *(_QWORD *)(v65 + 8) = 0;
      *(_QWORD *)(v65 + 16) = v65;
      *(_QWORD *)(v65 + 24) = v65;
      *(_QWORD *)(v65 + 32) = 0;
      if ( *(_QWORD *)(v64 + 8) )
      {
        *(_DWORD *)v65 = *(_DWORD *)v64;
        v68 = *(_QWORD *)(v64 + 8);
        *(_QWORD *)(v65 + 8) = v68;
        *(_QWORD *)(v65 + 16) = *(_QWORD *)(v64 + 16);
        *(_QWORD *)(v65 + 24) = *(_QWORD *)(v64 + 24);
        *(_QWORD *)(v68 + 8) = v65;
        *(_QWORD *)(v65 + 32) = *(_QWORD *)(v64 + 32);
        *(_QWORD *)(v64 + 8) = 0;
        *(_QWORD *)(v64 + 16) = v64;
        *(_QWORD *)(v64 + 24) = v64;
        *(_QWORD *)(v64 + 32) = 0;
      }
      v65 -= 48;
      v64 -= 48;
      --v63;
    }
    while ( v63 );
  }
}

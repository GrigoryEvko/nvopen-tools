// Function: sub_18790B0
// Address: 0x18790b0
//
__int64 __fastcall sub_18790B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rbx
  int *v10; // r10
  __int64 v11; // r11
  __int64 v12; // r11
  int v13; // ecx
  __int64 result; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // r13
  __int64 v18; // r14
  __int64 j; // r12
  __int64 v20; // rbx
  __int64 v21; // r15
  __int64 v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // r12
  __int64 v25; // r15
  __int64 i; // r13
  __int64 v27; // rbx
  __int64 v28; // rdi
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rdi
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rcx
  unsigned __int64 v35; // r13
  int *v36; // rbx
  __int64 v37; // r14
  __int64 v38; // r12
  __int64 v39; // rdi
  __int64 v40; // rax
  __int64 v41; // rax
  int *v42; // r10
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // r13
  int *v45; // rbx
  int *v46; // r14
  __int64 v47; // r12
  __int64 v48; // rdi
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rcx
  unsigned __int64 v53; // r12
  __int64 v54; // r15
  __int64 v55; // r13
  __int64 v56; // rbx
  __int64 v57; // rdi
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // r14
  __int64 v61; // rbx
  __int64 v62; // r12
  __int64 v63; // r13
  __int64 v64; // rdi
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // [rsp+8h] [rbp-88h]
  __int64 v68; // [rsp+10h] [rbp-80h]
  __int64 v69; // [rsp+10h] [rbp-80h]
  __int64 v70; // [rsp+18h] [rbp-78h]
  int *v71; // [rsp+18h] [rbp-78h]
  __int64 v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+20h] [rbp-70h]
  __int64 v74; // [rsp+20h] [rbp-70h]
  __int64 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+28h] [rbp-68h]
  int *v77; // [rsp+30h] [rbp-60h]
  __int64 v78; // [rsp+38h] [rbp-58h]
  __int64 v79; // [rsp+40h] [rbp-50h]
  __int64 v80; // [rsp+40h] [rbp-50h]
  __int64 v81; // [rsp+48h] [rbp-48h]
  __int64 v82; // [rsp+50h] [rbp-40h]
  __int64 v83; // [rsp+58h] [rbp-38h]
  __int64 v84; // [rsp+58h] [rbp-38h]

  while ( 1 )
  {
    v7 = a6;
    v82 = a1;
    v83 = a3;
    v8 = a7;
    if ( a5 <= a7 )
      v8 = a5;
    if ( v8 >= a4 )
    {
      v24 = a2;
      result = sub_1876230(a1, a2, a6);
      v25 = result;
      for ( i = a1 + 8; v25 != v7; i += 80 )
      {
        if ( v24 == v83 )
          return sub_1876230(v7, v25, i - 8);
        v27 = *(_QWORD *)(i + 8);
        result = *(_QWORD *)(v7 + 48);
        if ( *(_QWORD *)(v24 + 48) <= (unsigned __int64)result )
        {
          for ( ; v27; result = j_j___libc_free_0(v31, 40) )
          {
            sub_1876060(*(_QWORD *)(v27 + 24));
            v31 = v27;
            v27 = *(_QWORD *)(v27 + 16);
          }
          *(_QWORD *)(i + 8) = 0;
          *(_QWORD *)(i + 16) = i;
          *(_QWORD *)(i + 24) = i;
          *(_QWORD *)(i + 32) = 0;
          if ( *(_QWORD *)(v7 + 16) )
          {
            *(_DWORD *)i = *(_DWORD *)(v7 + 8);
            v32 = *(_QWORD *)(v7 + 16);
            *(_QWORD *)(i + 8) = v32;
            *(_QWORD *)(i + 16) = *(_QWORD *)(v7 + 24);
            *(_QWORD *)(i + 24) = *(_QWORD *)(v7 + 32);
            *(_QWORD *)(v32 + 8) = i;
            *(_QWORD *)(i + 32) = *(_QWORD *)(v7 + 40);
            *(_QWORD *)(v7 + 16) = 0;
            *(_QWORD *)(v7 + 24) = v7 + 8;
            *(_QWORD *)(v7 + 32) = v7 + 8;
            *(_QWORD *)(v7 + 40) = 0;
          }
          v33 = *(_QWORD *)(v7 + 48);
          v7 += 80;
          *(_QWORD *)(i + 40) = v33;
          *(_QWORD *)(i + 48) = *(_QWORD *)(v7 - 24);
          *(_QWORD *)(i + 56) = *(_QWORD *)(v7 - 16);
          *(_QWORD *)(i + 64) = *(_QWORD *)(v7 - 8);
        }
        else
        {
          for ( ; v27; result = j_j___libc_free_0(v28, 40) )
          {
            sub_1876060(*(_QWORD *)(v27 + 24));
            v28 = v27;
            v27 = *(_QWORD *)(v27 + 16);
          }
          *(_QWORD *)(i + 8) = 0;
          *(_QWORD *)(i + 16) = i;
          *(_QWORD *)(i + 24) = i;
          *(_QWORD *)(i + 32) = 0;
          if ( *(_QWORD *)(v24 + 16) )
          {
            *(_DWORD *)i = *(_DWORD *)(v24 + 8);
            v29 = *(_QWORD *)(v24 + 16);
            *(_QWORD *)(i + 8) = v29;
            *(_QWORD *)(i + 16) = *(_QWORD *)(v24 + 24);
            *(_QWORD *)(i + 24) = *(_QWORD *)(v24 + 32);
            *(_QWORD *)(v29 + 8) = i;
            *(_QWORD *)(i + 32) = *(_QWORD *)(v24 + 40);
            *(_QWORD *)(v24 + 16) = 0;
            *(_QWORD *)(v24 + 24) = v24 + 8;
            *(_QWORD *)(v24 + 32) = v24 + 8;
            *(_QWORD *)(v24 + 40) = 0;
          }
          v30 = *(_QWORD *)(v24 + 48);
          v24 += 80;
          *(_QWORD *)(i + 40) = v30;
          *(_QWORD *)(i + 48) = *(_QWORD *)(v24 - 24);
          *(_QWORD *)(i + 56) = *(_QWORD *)(v24 - 16);
          *(_QWORD *)(i + 64) = *(_QWORD *)(v24 - 8);
        }
      }
      return result;
    }
    v9 = a5;
    if ( a5 <= a7 )
      break;
    if ( a5 >= a4 )
    {
      v78 = a5 / 2;
      v77 = (int *)(a2 + 80 * (a5 / 2));
      v81 = sub_18739E0(a1, a2, (__int64)v77);
      v79 = 0xCCCCCCCCCCCCCCCDLL * ((v81 - a1) >> 4);
    }
    else
    {
      v79 = a4 / 2;
      v81 = a1 + 80 * (a4 / 2);
      v77 = (int *)sub_1873890(a2, a3, v81);
      v78 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v77 - (char *)v10) >> 4);
    }
    v76 = v11 - v79;
    if ( v11 - v79 <= v78 || a7 < v78 )
    {
      if ( a7 < v76 )
      {
        v12 = (__int64)sub_1878000(v81, v10, v77);
      }
      else
      {
        v12 = (__int64)v77;
        if ( v76 )
        {
          v70 = (__int64)v10;
          v73 = sub_1876230(v81, (__int64)v10, v7);
          sub_1876230(v70, (__int64)v77, v81);
          v34 = v73;
          v12 = (__int64)v77;
          v74 = v73 - v7;
          if ( v74 > 0 )
          {
            v68 = v9;
            v35 = 0xCCCCCCCCCCCCCCCDLL * (v74 >> 4);
            v67 = v7;
            v36 = v77 - 18;
            v37 = v34 - 72;
            do
            {
              v38 = *((_QWORD *)v36 + 1);
              while ( v38 )
              {
                sub_1876060(*(_QWORD *)(v38 + 24));
                v39 = v38;
                v38 = *(_QWORD *)(v38 + 16);
                j_j___libc_free_0(v39, 40);
              }
              *((_QWORD *)v36 + 1) = 0;
              *((_QWORD *)v36 + 2) = v36;
              *((_QWORD *)v36 + 3) = v36;
              *((_QWORD *)v36 + 4) = 0;
              if ( *(_QWORD *)(v37 + 8) )
              {
                *v36 = *(_DWORD *)v37;
                v40 = *(_QWORD *)(v37 + 8);
                *((_QWORD *)v36 + 1) = v40;
                *((_QWORD *)v36 + 2) = *(_QWORD *)(v37 + 16);
                *((_QWORD *)v36 + 3) = *(_QWORD *)(v37 + 24);
                *(_QWORD *)(v40 + 8) = v36;
                *((_QWORD *)v36 + 4) = *(_QWORD *)(v37 + 32);
                *(_QWORD *)(v37 + 8) = 0;
                *(_QWORD *)(v37 + 16) = v37;
                *(_QWORD *)(v37 + 24) = v37;
                *(_QWORD *)(v37 + 32) = 0;
              }
              v41 = *(_QWORD *)(v37 + 40);
              v36 -= 20;
              v37 -= 80;
              *((_QWORD *)v36 + 15) = v41;
              *((_QWORD *)v36 + 16) = *(_QWORD *)(v37 + 128);
              *((_QWORD *)v36 + 17) = *(_QWORD *)(v37 + 136);
              *((_QWORD *)v36 + 18) = *(_QWORD *)(v37 + 144);
              --v35;
            }
            while ( v35 );
            v9 = v68;
            v7 = v67;
            v12 = (__int64)&v77[-4 * (v74 >> 4)];
          }
        }
      }
    }
    else
    {
      v12 = v81;
      if ( v78 )
      {
        v71 = v10;
        v75 = sub_1876230((__int64)v10, (__int64)v77, v7);
        v42 = v71 - 18;
        v43 = 0xCCCCCCCCCCCCCCCDLL * (((__int64)v71 - v81) >> 4);
        if ( (__int64)v71 - v81 > 0 )
        {
          v72 = v9;
          v44 = v43;
          v45 = v77 - 18;
          v69 = v7;
          v46 = v42;
          do
          {
            v47 = *((_QWORD *)v45 + 1);
            while ( v47 )
            {
              sub_1876060(*(_QWORD *)(v47 + 24));
              v48 = v47;
              v47 = *(_QWORD *)(v47 + 16);
              j_j___libc_free_0(v48, 40);
            }
            *((_QWORD *)v45 + 1) = 0;
            *((_QWORD *)v45 + 2) = v45;
            *((_QWORD *)v45 + 3) = v45;
            *((_QWORD *)v45 + 4) = 0;
            if ( *((_QWORD *)v46 + 1) )
            {
              *v45 = *v46;
              v49 = *((_QWORD *)v46 + 1);
              *((_QWORD *)v45 + 1) = v49;
              *((_QWORD *)v45 + 2) = *((_QWORD *)v46 + 2);
              *((_QWORD *)v45 + 3) = *((_QWORD *)v46 + 3);
              *(_QWORD *)(v49 + 8) = v45;
              *((_QWORD *)v45 + 4) = *((_QWORD *)v46 + 4);
              *((_QWORD *)v46 + 1) = 0;
              *((_QWORD *)v46 + 2) = v46;
              *((_QWORD *)v46 + 3) = v46;
              *((_QWORD *)v46 + 4) = 0;
            }
            v50 = *((_QWORD *)v46 + 5);
            v45 -= 20;
            v46 -= 20;
            *((_QWORD *)v45 + 15) = v50;
            *((_QWORD *)v45 + 16) = *((_QWORD *)v46 + 16);
            *((_QWORD *)v45 + 17) = *((_QWORD *)v46 + 17);
            *((_QWORD *)v45 + 18) = *((_QWORD *)v46 + 18);
            --v44;
          }
          while ( v44 );
          v9 = v72;
          v7 = v69;
        }
        v12 = sub_1876230(v7, v75, v81);
      }
    }
    v13 = v79;
    v80 = v12;
    sub_18790B0(v82, v81, v12, v13, v78, v7, a7);
    a6 = v7;
    a4 = v76;
    a2 = (__int64)v77;
    a5 = v9 - v78;
    a1 = v80;
    a3 = v83;
  }
  result = sub_1876230(a2, a3, a6);
  v15 = result;
  if ( a1 != a2 )
  {
    if ( v7 == result )
      return result;
    v16 = v83;
    v84 = v7;
    v17 = result - 80;
    v18 = a2 - 80;
    for ( j = v16 - 72; ; j -= 80 )
    {
      v20 = *(_QWORD *)(j + 8);
      result = *(_QWORD *)(v18 + 48);
      v21 = j - 8;
      if ( *(_QWORD *)(v17 + 48) <= (unsigned __int64)result )
      {
        for ( ; v20; result = j_j___libc_free_0(v51, 40) )
        {
          sub_1876060(*(_QWORD *)(v20 + 24));
          v51 = v20;
          v20 = *(_QWORD *)(v20 + 16);
        }
        *(_QWORD *)(j + 8) = 0;
        *(_QWORD *)(j + 16) = j;
        *(_QWORD *)(j + 24) = j;
        *(_QWORD *)(j + 32) = 0;
        if ( *(_QWORD *)(v17 + 16) )
        {
          *(_DWORD *)j = *(_DWORD *)(v17 + 8);
          v52 = *(_QWORD *)(v17 + 16);
          *(_QWORD *)(j + 8) = v52;
          *(_QWORD *)(j + 16) = *(_QWORD *)(v17 + 24);
          *(_QWORD *)(j + 24) = *(_QWORD *)(v17 + 32);
          *(_QWORD *)(v52 + 8) = j;
          *(_QWORD *)(j + 32) = *(_QWORD *)(v17 + 40);
          *(_QWORD *)(v17 + 16) = 0;
          *(_QWORD *)(v17 + 24) = v17 + 8;
          *(_QWORD *)(v17 + 32) = v17 + 8;
          *(_QWORD *)(v17 + 40) = 0;
        }
        *(_QWORD *)(j + 40) = *(_QWORD *)(v17 + 48);
        *(_QWORD *)(j + 48) = *(_QWORD *)(v17 + 56);
        *(_QWORD *)(j + 56) = *(_QWORD *)(v17 + 64);
        *(_QWORD *)(j + 64) = *(_QWORD *)(v17 + 72);
        if ( v84 == v17 )
          return result;
        v17 -= 80;
      }
      else
      {
        while ( v20 )
        {
          sub_1876060(*(_QWORD *)(v20 + 24));
          v22 = v20;
          v20 = *(_QWORD *)(v20 + 16);
          j_j___libc_free_0(v22, 40);
        }
        *(_QWORD *)(j + 8) = 0;
        *(_QWORD *)(j + 16) = j;
        *(_QWORD *)(j + 24) = j;
        *(_QWORD *)(j + 32) = 0;
        if ( *(_QWORD *)(v18 + 16) )
        {
          *(_DWORD *)j = *(_DWORD *)(v18 + 8);
          v23 = *(_QWORD *)(v18 + 16);
          *(_QWORD *)(j + 8) = v23;
          *(_QWORD *)(j + 16) = *(_QWORD *)(v18 + 24);
          *(_QWORD *)(j + 24) = *(_QWORD *)(v18 + 32);
          *(_QWORD *)(v23 + 8) = j;
          *(_QWORD *)(j + 32) = *(_QWORD *)(v18 + 40);
          *(_QWORD *)(v18 + 16) = 0;
          *(_QWORD *)(v18 + 24) = v18 + 8;
          *(_QWORD *)(v18 + 32) = v18 + 8;
          *(_QWORD *)(v18 + 40) = 0;
        }
        *(_QWORD *)(j + 40) = *(_QWORD *)(v18 + 48);
        *(_QWORD *)(j + 48) = *(_QWORD *)(v18 + 56);
        *(_QWORD *)(j + 56) = *(_QWORD *)(v18 + 64);
        *(_QWORD *)(j + 64) = *(_QWORD *)(v18 + 72);
        if ( v18 == v82 )
        {
          result = v17 + 80 - v84;
          v53 = 0xCCCCCCCCCCCCCCCDLL * (result >> 4);
          if ( result > 0 )
          {
            v54 = v21 - 72;
            v55 = v17 + 8;
            do
            {
              v56 = *(_QWORD *)(v54 + 8);
              while ( v56 )
              {
                sub_1876060(*(_QWORD *)(v56 + 24));
                v57 = v56;
                v56 = *(_QWORD *)(v56 + 16);
                j_j___libc_free_0(v57, 40);
              }
              *(_QWORD *)(v54 + 8) = 0;
              *(_QWORD *)(v54 + 16) = v54;
              *(_QWORD *)(v54 + 24) = v54;
              *(_QWORD *)(v54 + 32) = 0;
              if ( *(_QWORD *)(v55 + 8) )
              {
                *(_DWORD *)v54 = *(_DWORD *)v55;
                v58 = *(_QWORD *)(v55 + 8);
                *(_QWORD *)(v54 + 8) = v58;
                *(_QWORD *)(v54 + 16) = *(_QWORD *)(v55 + 16);
                *(_QWORD *)(v54 + 24) = *(_QWORD *)(v55 + 24);
                *(_QWORD *)(v58 + 8) = v54;
                *(_QWORD *)(v54 + 32) = *(_QWORD *)(v55 + 32);
                *(_QWORD *)(v55 + 8) = 0;
                *(_QWORD *)(v55 + 16) = v55;
                *(_QWORD *)(v55 + 24) = v55;
                *(_QWORD *)(v55 + 32) = 0;
              }
              v59 = *(_QWORD *)(v55 + 40);
              v54 -= 80;
              v55 -= 80;
              *(_QWORD *)(v54 + 120) = v59;
              *(_QWORD *)(v54 + 128) = *(_QWORD *)(v55 + 128);
              *(_QWORD *)(v54 + 136) = *(_QWORD *)(v55 + 136);
              result = *(_QWORD *)(v55 + 144);
              *(_QWORD *)(v54 + 144) = result;
              --v53;
            }
            while ( v53 );
          }
          return result;
        }
        v18 -= 80;
      }
    }
  }
  result -= v7;
  v60 = 0xCCCCCCCCCCCCCCCDLL * (result >> 4);
  if ( result > 0 )
  {
    v61 = v15 - 72;
    v62 = v83 - 72;
    do
    {
      v63 = *(_QWORD *)(v62 + 8);
      while ( v63 )
      {
        sub_1876060(*(_QWORD *)(v63 + 24));
        v64 = v63;
        v63 = *(_QWORD *)(v63 + 16);
        j_j___libc_free_0(v64, 40);
      }
      *(_QWORD *)(v62 + 8) = 0;
      *(_QWORD *)(v62 + 16) = v62;
      *(_QWORD *)(v62 + 24) = v62;
      *(_QWORD *)(v62 + 32) = 0;
      if ( *(_QWORD *)(v61 + 8) )
      {
        *(_DWORD *)v62 = *(_DWORD *)v61;
        v65 = *(_QWORD *)(v61 + 8);
        *(_QWORD *)(v62 + 8) = v65;
        *(_QWORD *)(v62 + 16) = *(_QWORD *)(v61 + 16);
        *(_QWORD *)(v62 + 24) = *(_QWORD *)(v61 + 24);
        *(_QWORD *)(v65 + 8) = v62;
        *(_QWORD *)(v62 + 32) = *(_QWORD *)(v61 + 32);
        *(_QWORD *)(v61 + 8) = 0;
        *(_QWORD *)(v61 + 16) = v61;
        *(_QWORD *)(v61 + 24) = v61;
        *(_QWORD *)(v61 + 32) = 0;
      }
      v66 = *(_QWORD *)(v61 + 40);
      v62 -= 80;
      v61 -= 80;
      *(_QWORD *)(v62 + 120) = v66;
      *(_QWORD *)(v62 + 128) = *(_QWORD *)(v61 + 128);
      *(_QWORD *)(v62 + 136) = *(_QWORD *)(v61 + 136);
      result = *(_QWORD *)(v61 + 144);
      *(_QWORD *)(v62 + 144) = result;
      --v60;
    }
    while ( v60 );
  }
  return result;
}

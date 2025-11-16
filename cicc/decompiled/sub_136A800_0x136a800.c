// Function: sub_136A800
// Address: 0x136a800
//
unsigned __int64 __fastcall sub_136A800(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v3; // rbx
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  __int64 *v6; // r14
  __int64 *v7; // rbx
  __int64 v8; // r13
  unsigned __int64 *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rdx
  __int64 *v12; // rdx
  __int64 **v13; // r13
  __int64 v14; // r15
  int v15; // edx
  int v16; // r14d
  __int64 v17; // rbx
  int v18; // edx
  __int64 v19; // r9
  __int64 v20; // rdi
  unsigned int v21; // esi
  __int64 *v22; // rax
  __int64 v23; // r11
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // r15
  __int64 *v30; // r13
  __int64 v31; // r14
  __int64 v32; // rbx
  unsigned __int64 *v33; // rdx
  _QWORD *v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rcx
  unsigned __int64 v37; // rbx
  __int64 v38; // rdx
  int v39; // eax
  __int64 v40; // rcx
  __int64 v41; // rsi
  int v42; // edi
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // r9
  __int64 v46; // rdx
  int v47; // ecx
  __int64 v48; // rax
  int v49; // ecx
  __int64 v50; // rdi
  __int64 v51; // rsi
  unsigned int v52; // edx
  __int64 *v53; // rax
  __int64 v54; // r9
  __int64 *v55; // r13
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // r14
  __int64 **v59; // r15
  __int64 v60; // rax
  __int64 *v61; // rdi
  bool v62; // al
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned __int64 *v65; // rbx
  unsigned __int64 *v66; // r12
  __int64 v67; // rdi
  __int64 v68; // rdx
  int v69; // eax
  int v70; // eax
  int v71; // r10d
  int v72; // eax
  int v73; // r10d
  int v74; // r10d
  __int64 v75; // [rsp+0h] [rbp-90h]
  __int64 v76; // [rsp+8h] [rbp-88h]
  __int64 v77; // [rsp+10h] [rbp-80h] BYREF
  __int64 v78; // [rsp+18h] [rbp-78h]
  __int64 *v79; // [rsp+20h] [rbp-70h]
  unsigned __int64 v80; // [rsp+28h] [rbp-68h]
  __int64 v81; // [rsp+30h] [rbp-60h]
  unsigned __int64 *v82; // [rsp+38h] [rbp-58h]
  _QWORD *v83; // [rsp+40h] [rbp-50h]
  unsigned __int64 v84; // [rsp+48h] [rbp-48h]
  __int64 v85; // [rsp+50h] [rbp-40h]
  unsigned __int64 *v86; // [rsp+58h] [rbp-38h]

  result = *(_QWORD *)(a1 + 120);
  if ( *(_QWORD *)(result + 32) == *(_QWORD *)(result + 40) )
    return result;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v78 = 8;
  v77 = sub_22077B0(64);
  v3 = (unsigned __int64 *)(v77 + 24);
  result = sub_22077B0(512);
  v4 = *(_QWORD *)(a1 + 120);
  v82 = (unsigned __int64 *)(v77 + 24);
  v5 = result + 512;
  *(_QWORD *)(v77 + 24) = result;
  v86 = v3;
  v84 = result;
  v85 = result + 512;
  v83 = (_QWORD *)result;
  v6 = *(__int64 **)(v4 + 40);
  v81 = result + 512;
  v7 = *(__int64 **)(v4 + 32);
  v80 = result;
  v79 = (__int64 *)result;
  if ( v6 == v7 )
    goto LABEL_35;
  while ( 1 )
  {
    v8 = *v7;
    if ( result == v5 - 16 )
      break;
    if ( result )
    {
      *(_QWORD *)result = v8;
      *(_QWORD *)(result + 8) = 0;
      result = (unsigned __int64)v83;
    }
    result += 16LL;
    ++v7;
    v83 = (_QWORD *)result;
    if ( v6 == v7 )
      goto LABEL_15;
LABEL_7:
    v5 = v85;
  }
  v9 = v86;
  if ( 32 * (v86 - v82 - 1) + ((__int64)(result - v84) >> 4) + ((v81 - (__int64)v79) >> 4) == 0x7FFFFFFFFFFFFFFLL )
LABEL_78:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( (unsigned __int64)(v78 - (((__int64)v86 - v77) >> 3)) <= 1 )
  {
    sub_136A680(&v77, 1u, 0);
    v9 = v86;
  }
  v9[1] = sub_22077B0(512);
  v10 = v83;
  if ( v83 )
  {
    *v83 = v8;
    v10[1] = 0;
  }
  ++v7;
  result = *++v86;
  v11 = *v86 + 512;
  v84 = result;
  v85 = v11;
  v83 = (_QWORD *)result;
  if ( v6 != v7 )
    goto LABEL_7;
LABEL_15:
  v12 = v79;
  if ( v79 != (__int64 *)result )
  {
    v75 = a1 + 88;
    do
    {
      v13 = (__int64 **)*v12;
      v14 = v12[1];
      if ( v12 == (__int64 *)(v81 - 16) )
      {
        j_j___libc_free_0(v80, 512);
        v68 = *++v82 + 512;
        v80 = *v82;
        v81 = v68;
        v79 = (__int64 *)v80;
      }
      else
      {
        v79 += 2;
      }
      v15 = *(_DWORD *)(a1 + 184);
      v16 = -1;
      v17 = 0x17FFFFFFE8LL;
      if ( v15 )
      {
        v18 = v15 - 1;
        v19 = *(_QWORD *)(a1 + 168);
        v20 = *v13[4];
        v21 = v18 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
        v22 = (__int64 *)(v19 + 16LL * v21);
        v23 = *v22;
        if ( v20 == *v22 )
        {
LABEL_21:
          v24 = *((unsigned int *)v22 + 2);
          v16 = v24;
          v17 = 24 * v24;
        }
        else
        {
          v69 = 1;
          while ( v23 != -8 )
          {
            v73 = v69 + 1;
            v21 = v18 & (v69 + v21);
            v22 = (__int64 *)(v19 + 16LL * v21);
            v23 = *v22;
            if ( v20 == *v22 )
              goto LABEL_21;
            v69 = v73;
          }
          v17 = 0x17FFFFFFE8LL;
          v16 = -1;
        }
      }
      v25 = sub_22077B0(192);
      *(_QWORD *)(v25 + 16) = v14;
      v26 = v25;
      v25 += 48;
      *(_BYTE *)(v25 - 24) = 0;
      *(_DWORD *)(v25 - 20) = 1;
      *(_QWORD *)(v26 + 32) = v25;
      *(_QWORD *)(v26 + 40) = 0x400000000LL;
      *(_QWORD *)(v26 + 112) = v26 + 128;
      *(_QWORD *)(v26 + 120) = 0x400000001LL;
      *(_QWORD *)(v26 + 144) = v26 + 160;
      *(_QWORD *)(v26 + 152) = 0x100000001LL;
      *(_WORD *)(v26 + 184) = 0;
      *(_DWORD *)(v26 + 128) = v16;
      *(_QWORD *)(v26 + 160) = 0;
      *(_QWORD *)(v26 + 168) = 0;
      *(_QWORD *)(v26 + 176) = 0;
      sub_2208C80(v26, v75);
      v27 = *(_QWORD *)(a1 + 96);
      v28 = *(_QWORD *)(a1 + 64);
      ++*(_QWORD *)(a1 + 104);
      *(_QWORD *)(v28 + v17 + 8) = v27 + 16;
      v29 = v13[1];
      v30 = v13[2];
      for ( result = (unsigned __int64)v83; v30 != v29; v83 = (_QWORD *)result )
      {
        while ( 1 )
        {
          v31 = *v29;
          v32 = *(_QWORD *)(a1 + 96) + 16LL;
          if ( result == v85 - 16 )
            break;
          if ( result )
          {
            *(_QWORD *)result = v31;
            *(_QWORD *)(result + 8) = v32;
            result = (unsigned __int64)v83;
          }
          result += 16LL;
          ++v29;
          v83 = (_QWORD *)result;
          if ( v30 == v29 )
            goto LABEL_34;
        }
        v33 = v86;
        if ( ((__int64)(result - v84) >> 4) + 32 * (v86 - v82 - 1) + ((v81 - (__int64)v79) >> 4) == 0x7FFFFFFFFFFFFFFLL )
          goto LABEL_78;
        if ( (unsigned __int64)(v78 - (((__int64)v86 - v77) >> 3)) <= 1 )
        {
          sub_136A680(&v77, 1u, 0);
          v33 = v86;
        }
        v33[1] = sub_22077B0(512);
        v34 = v83;
        if ( v83 )
        {
          *v83 = v31;
          v34[1] = v32;
        }
        ++v29;
        result = *++v86;
        v35 = *v86 + 512;
        v84 = result;
        v85 = v35;
      }
LABEL_34:
      v12 = v79;
    }
    while ( v79 != (__int64 *)result );
  }
LABEL_35:
  v36 = *(_QWORD *)(a1 + 136);
  v37 = 0;
  if ( *(_QWORD *)(a1 + 144) != v36 )
  {
    do
    {
      v57 = *(_QWORD *)(a1 + 64);
      v58 = v57 + 24 * v37;
      v59 = *(__int64 ***)(v58 + 8);
      if ( v59 )
      {
        v60 = *((unsigned int *)v59 + 3);
        v61 = v59[12];
        if ( (unsigned int)v60 <= 1 )
        {
          if ( *(_DWORD *)v58 == *(_DWORD *)v61 )
            goto LABEL_52;
        }
        else
        {
          v76 = v36;
          v62 = sub_1369030(v61, (_DWORD *)v61 + v60, (_DWORD *)(v57 + 24 * v37));
          v36 = v76;
          if ( v62 )
          {
LABEL_52:
            v55 = *v59;
            if ( *v59 )
            {
              v63 = *((unsigned int *)v55 + 3);
              if ( (unsigned int)v63 <= 1 )
                goto LABEL_45;
              if ( !sub_1369030((_DWORD *)v55[12], (_DWORD *)(v55[12] + 4 * v63), (_DWORD *)v58) )
                goto LABEL_45;
              v55 = (__int64 *)*v55;
              if ( v55 )
                goto LABEL_45;
            }
            goto LABEL_48;
          }
        }
      }
      v38 = *(_QWORD *)(a1 + 120);
      v39 = *(_DWORD *)(v38 + 24);
      if ( v39 )
      {
        v40 = *(_QWORD *)(v36 + 8 * v37);
        v41 = *(_QWORD *)(v38 + 8);
        v42 = v39 - 1;
        v43 = (v39 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v44 = (__int64 *)(v41 + 16LL * v43);
        v45 = *v44;
        if ( v40 != *v44 )
        {
          v70 = 1;
          while ( v45 != -8 )
          {
            v71 = v70 + 1;
            v43 = v42 & (v70 + v43);
            v44 = (__int64 *)(v41 + 16LL * v43);
            v45 = *v44;
            if ( v40 == *v44 )
              goto LABEL_40;
            v70 = v71;
          }
          goto LABEL_48;
        }
LABEL_40:
        v46 = v44[1];
        if ( v46 )
        {
          v47 = *(_DWORD *)(a1 + 184);
          v48 = 0x17FFFFFFE8LL;
          if ( v47 )
          {
            v49 = v47 - 1;
            v50 = *(_QWORD *)(a1 + 168);
            v51 = **(_QWORD **)(v46 + 32);
            v52 = v49 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
            v53 = (__int64 *)(v50 + 16LL * v52);
            v54 = *v53;
            if ( v51 == *v53 )
            {
LABEL_43:
              v48 = 24LL * *((unsigned int *)v53 + 2);
            }
            else
            {
              v72 = 1;
              while ( v54 != -8 )
              {
                v74 = v72 + 1;
                v52 = v49 & (v72 + v52);
                v53 = (__int64 *)(v50 + 16LL * v52);
                v54 = *v53;
                if ( v51 == *v53 )
                  goto LABEL_43;
                v72 = v74;
              }
              v48 = 0x17FFFFFFE8LL;
            }
          }
          v55 = *(__int64 **)(v57 + v48 + 8);
          *(_QWORD *)(v58 + 8) = v55;
LABEL_45:
          v56 = *((unsigned int *)v55 + 26);
          if ( (unsigned int)v56 >= *((_DWORD *)v55 + 27) )
          {
            sub_16CD150(v55 + 12, v55 + 14, 0, 4);
            v56 = *((unsigned int *)v55 + 26);
          }
          *(_DWORD *)(v55[12] + 4 * v56) = v37;
          ++*((_DWORD *)v55 + 26);
        }
      }
LABEL_48:
      v36 = *(_QWORD *)(a1 + 136);
      ++v37;
      result = (*(_QWORD *)(a1 + 144) - v36) >> 3;
    }
    while ( v37 < result );
  }
  v64 = v77;
  if ( v77 )
  {
    v65 = v82;
    v66 = v86 + 1;
    if ( v86 + 1 > v82 )
    {
      do
      {
        v67 = *v65++;
        j_j___libc_free_0(v67, 512);
      }
      while ( v66 > v65 );
      v64 = v77;
    }
    return j_j___libc_free_0(v64, 8 * v78);
  }
  return result;
}

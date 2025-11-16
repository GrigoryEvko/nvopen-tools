// Function: sub_1DDE360
// Address: 0x1dde360
//
unsigned __int64 __fastcall sub_1DDE360(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 *v3; // rbx
  int v4; // r8d
  __int64 v5; // r9
  __int64 v6; // rcx
  unsigned __int64 v7; // rdx
  __int64 *v8; // r14
  __int64 *v9; // rbx
  __int64 v10; // r13
  unsigned __int64 *v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rdx
  __int64 **v15; // r13
  __int64 v16; // r15
  int v17; // edx
  int v18; // r14d
  __int64 v19; // rbx
  int v20; // edx
  __int64 v21; // r9
  __int64 v22; // rdi
  unsigned int v23; // esi
  __int64 *v24; // rax
  __int64 v25; // r11
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 *v31; // r15
  __int64 *v32; // r13
  __int64 v33; // r14
  __int64 v34; // rbx
  unsigned __int64 *v35; // rdx
  _QWORD *v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int64 v39; // rbx
  __int64 v40; // rdx
  int v41; // eax
  __int64 v42; // rcx
  __int64 v43; // rsi
  int v44; // edi
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // rdx
  int v48; // ecx
  __int64 v49; // rax
  int v50; // ecx
  __int64 v51; // rdi
  __int64 v52; // rsi
  unsigned int v53; // edx
  __int64 *v54; // rax
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
  if ( *(_QWORD *)(result + 264) == *(_QWORD *)(result + 272) )
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
  v6 = *(_QWORD *)(a1 + 120);
  v82 = (unsigned __int64 *)(v77 + 24);
  *(_QWORD *)(v77 + 24) = result;
  v7 = result + 512;
  v86 = v3;
  v8 = *(__int64 **)(v6 + 272);
  v9 = *(__int64 **)(v6 + 264);
  v80 = result;
  v81 = result + 512;
  v84 = result;
  v85 = result + 512;
  v79 = (__int64 *)result;
  v83 = (_QWORD *)result;
  if ( v8 == v9 )
    goto LABEL_35;
  while ( 1 )
  {
    v10 = *v9;
    if ( result == v7 - 16 )
      break;
    if ( result )
    {
      *(_QWORD *)result = v10;
      *(_QWORD *)(result + 8) = 0;
      result = (unsigned __int64)v83;
    }
    result += 16LL;
    ++v9;
    v83 = (_QWORD *)result;
    if ( v8 == v9 )
      goto LABEL_15;
LABEL_7:
    v7 = v85;
  }
  v11 = v86;
  if ( 32 * (v86 - v82 - 1) + ((__int64)(result - v84) >> 4) + ((v81 - (__int64)v79) >> 4) == 0x7FFFFFFFFFFFFFFLL )
LABEL_78:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  if ( (unsigned __int64)(v78 - (((__int64)v86 - v77) >> 3)) <= 1 )
  {
    sub_1DDE1E0(&v77, 1u, 0);
    v11 = v86;
  }
  v11[1] = sub_22077B0(512);
  v12 = v83;
  if ( v83 )
  {
    *v83 = v10;
    v12[1] = 0;
  }
  ++v9;
  result = *++v86;
  v13 = *v86 + 512;
  v84 = result;
  v85 = v13;
  v83 = (_QWORD *)result;
  if ( v8 != v9 )
    goto LABEL_7;
LABEL_15:
  v14 = v79;
  if ( v79 != (__int64 *)result )
  {
    v75 = a1 + 88;
    do
    {
      v15 = (__int64 **)*v14;
      v16 = v14[1];
      if ( v14 == (__int64 *)(v81 - 16) )
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
      v17 = *(_DWORD *)(a1 + 184);
      v18 = -1;
      v19 = 0x17FFFFFFE8LL;
      if ( v17 )
      {
        v20 = v17 - 1;
        v21 = *(_QWORD *)(a1 + 168);
        v22 = *v15[4];
        v23 = v20 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v24 = (__int64 *)(v21 + 16LL * v23);
        v25 = *v24;
        if ( v22 == *v24 )
        {
LABEL_21:
          v26 = *((unsigned int *)v24 + 2);
          v18 = v26;
          v19 = 24 * v26;
        }
        else
        {
          v69 = 1;
          while ( v25 != -8 )
          {
            v73 = v69 + 1;
            v23 = v20 & (v69 + v23);
            v24 = (__int64 *)(v21 + 16LL * v23);
            v25 = *v24;
            if ( v22 == *v24 )
              goto LABEL_21;
            v69 = v73;
          }
          v19 = 0x17FFFFFFE8LL;
          v18 = -1;
        }
      }
      v27 = sub_22077B0(192);
      *(_QWORD *)(v27 + 16) = v16;
      v28 = v27;
      v27 += 48;
      *(_BYTE *)(v27 - 24) = 0;
      *(_DWORD *)(v27 - 20) = 1;
      *(_QWORD *)(v28 + 32) = v27;
      *(_QWORD *)(v28 + 40) = 0x400000000LL;
      *(_QWORD *)(v28 + 112) = v28 + 128;
      *(_QWORD *)(v28 + 120) = 0x400000001LL;
      *(_QWORD *)(v28 + 144) = v28 + 160;
      *(_QWORD *)(v28 + 152) = 0x100000001LL;
      *(_WORD *)(v28 + 184) = 0;
      *(_DWORD *)(v28 + 128) = v18;
      *(_QWORD *)(v28 + 160) = 0;
      *(_QWORD *)(v28 + 168) = 0;
      *(_QWORD *)(v28 + 176) = 0;
      sub_2208C80(v28, v75);
      v29 = *(_QWORD *)(a1 + 96);
      v30 = *(_QWORD *)(a1 + 64);
      ++*(_QWORD *)(a1 + 104);
      *(_QWORD *)(v30 + v19 + 8) = v29 + 16;
      v31 = v15[1];
      v32 = v15[2];
      for ( result = (unsigned __int64)v83; v32 != v31; v83 = (_QWORD *)result )
      {
        while ( 1 )
        {
          v33 = *v31;
          v34 = *(_QWORD *)(a1 + 96) + 16LL;
          if ( result == v85 - 16 )
            break;
          if ( result )
          {
            *(_QWORD *)result = v33;
            *(_QWORD *)(result + 8) = v34;
            result = (unsigned __int64)v83;
          }
          result += 16LL;
          ++v31;
          v83 = (_QWORD *)result;
          if ( v32 == v31 )
            goto LABEL_34;
        }
        v35 = v86;
        if ( ((__int64)(result - v84) >> 4) + 32 * (v86 - v82 - 1) + ((v81 - (__int64)v79) >> 4) == 0x7FFFFFFFFFFFFFFLL )
          goto LABEL_78;
        if ( (unsigned __int64)(v78 - (((__int64)v86 - v77) >> 3)) <= 1 )
        {
          sub_1DDE1E0(&v77, 1u, 0);
          v35 = v86;
        }
        v35[1] = sub_22077B0(512);
        v36 = v83;
        if ( v83 )
        {
          *v83 = v33;
          v36[1] = v34;
        }
        ++v31;
        result = *++v86;
        v37 = *v86 + 512;
        v84 = result;
        v85 = v37;
      }
LABEL_34:
      v14 = v79;
    }
    while ( v79 != (__int64 *)result );
  }
LABEL_35:
  v38 = *(_QWORD *)(a1 + 136);
  v39 = 0;
  if ( *(_QWORD *)(a1 + 144) != v38 )
  {
    do
    {
      v57 = *(_QWORD *)(a1 + 64);
      v58 = v57 + 24 * v39;
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
          v76 = v38;
          v62 = sub_1369030(v61, (_DWORD *)v61 + v60, (_DWORD *)(v57 + 24 * v39));
          v38 = v76;
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
      v40 = *(_QWORD *)(a1 + 120);
      v41 = *(_DWORD *)(v40 + 256);
      if ( v41 )
      {
        v42 = *(_QWORD *)(v38 + 8 * v39);
        v43 = *(_QWORD *)(v40 + 240);
        v44 = v41 - 1;
        v45 = (v41 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v46 = (__int64 *)(v43 + 16LL * v45);
        v5 = *v46;
        if ( v42 != *v46 )
        {
          v70 = 1;
          while ( v5 != -8 )
          {
            v71 = v70 + 1;
            v45 = v44 & (v70 + v45);
            v46 = (__int64 *)(v43 + 16LL * v45);
            v5 = *v46;
            if ( v42 == *v46 )
              goto LABEL_40;
            v70 = v71;
          }
          goto LABEL_48;
        }
LABEL_40:
        v47 = v46[1];
        if ( v47 )
        {
          v48 = *(_DWORD *)(a1 + 184);
          v49 = 0x17FFFFFFE8LL;
          if ( v48 )
          {
            v50 = v48 - 1;
            v51 = *(_QWORD *)(a1 + 168);
            v52 = **(_QWORD **)(v47 + 32);
            v53 = v50 & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
            v54 = (__int64 *)(v51 + 16LL * v53);
            v5 = *v54;
            if ( v52 == *v54 )
            {
LABEL_43:
              v49 = 24LL * *((unsigned int *)v54 + 2);
            }
            else
            {
              v72 = 1;
              while ( v5 != -8 )
              {
                v74 = v72 + 1;
                v53 = v50 & (v72 + v53);
                v54 = (__int64 *)(v51 + 16LL * v53);
                v5 = *v54;
                if ( v52 == *v54 )
                  goto LABEL_43;
                v72 = v74;
              }
              v49 = 0x17FFFFFFE8LL;
            }
          }
          v55 = *(__int64 **)(v57 + v49 + 8);
          *(_QWORD *)(v58 + 8) = v55;
LABEL_45:
          v56 = *((unsigned int *)v55 + 26);
          if ( (unsigned int)v56 >= *((_DWORD *)v55 + 27) )
          {
            sub_16CD150((__int64)(v55 + 12), v55 + 14, 0, 4, v4, v5);
            v56 = *((unsigned int *)v55 + 26);
          }
          *(_DWORD *)(v55[12] + 4 * v56) = v39;
          ++*((_DWORD *)v55 + 26);
        }
      }
LABEL_48:
      v38 = *(_QWORD *)(a1 + 136);
      ++v39;
      result = (*(_QWORD *)(a1 + 144) - v38) >> 3;
    }
    while ( v39 < result );
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

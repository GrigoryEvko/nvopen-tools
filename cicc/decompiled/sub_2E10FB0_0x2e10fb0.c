// Function: sub_2E10FB0
// Address: 0x2e10fb0
//
__int64 __fastcall sub_2E10FB0(unsigned int *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rcx
  _DWORD *v11; // rax
  _DWORD *i; // rdx
  __int64 v13; // r14
  __int64 result; // rax
  __int64 v15; // rbx
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rcx
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  unsigned __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rbx
  __int64 v27; // r14
  __int64 v28; // r8
  unsigned __int64 v29; // rcx
  unsigned __int64 m; // rax
  __int64 n; // rsi
  __int16 v32; // dx
  __int64 v33; // rsi
  __int64 v34; // rdi
  unsigned int v35; // ecx
  __int64 *v36; // rdx
  __int64 v37; // r10
  __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  __int64 v41; // rax
  unsigned __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r13
  __int64 v46; // rdi
  unsigned __int64 v47; // rcx
  int v48; // r8d
  unsigned __int64 v49; // rdx
  int v50; // esi
  __int64 j; // rdx
  unsigned __int64 v52; // rax
  __int64 k; // rsi
  __int16 v54; // dx
  unsigned int v55; // esi
  __int64 v56; // r8
  unsigned int v57; // ecx
  __int64 *v58; // rdx
  __int64 v59; // r10
  __int64 v60; // rax
  unsigned __int64 v61; // rbx
  __int64 v62; // rax
  unsigned __int64 v63; // rcx
  int v64; // edx
  int v65; // edx
  __int64 v66; // rbx
  __int64 v67; // r8
  __int64 v68; // r13
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // rcx
  __int64 v72; // [rsp+8h] [rbp-58h]
  __int64 v73; // [rsp+10h] [rbp-50h]
  _DWORD *v74; // [rsp+18h] [rbp-48h]
  const void *v75; // [rsp+20h] [rbp-40h]
  __int64 v76; // [rsp+28h] [rbp-38h]
  unsigned __int64 v77; // [rsp+28h] [rbp-38h]

  v7 = *(_QWORD *)a1;
  v8 = a1[88];
  v9 = (__int64)(*(_QWORD *)(*(_QWORD *)a1 + 104LL) - *(_QWORD *)(*(_QWORD *)a1 + 96LL)) >> 3;
  if ( (unsigned int)v9 != v8 )
  {
    if ( (unsigned int)v9 < v8 )
    {
      a1[88] = v9;
    }
    else
    {
      if ( (unsigned int)v9 > (unsigned __int64)a1[89] )
      {
        sub_C8D5F0(
          (__int64)(a1 + 86),
          a1 + 90,
          (unsigned int)((__int64)(*(_QWORD *)(*(_QWORD *)a1 + 104LL) - *(_QWORD *)(*(_QWORD *)a1 + 96LL)) >> 3),
          8u,
          a5,
          a6);
        v8 = a1[88];
      }
      v10 = *((_QWORD *)a1 + 43);
      v11 = (_DWORD *)(v10 + 8 * v8);
      for ( i = (_DWORD *)(v10 + 8LL * (unsigned int)v9); i != v11; v11 += 2 )
      {
        if ( v11 )
        {
          *v11 = 0;
          v11[1] = 0;
        }
      }
      a1[88] = v9;
      v7 = *(_QWORD *)a1;
    }
  }
  v13 = *(_QWORD *)(v7 + 328);
  result = v7 + 320;
  v72 = result;
  v75 = a1 + 70;
  if ( result != v13 )
  {
    while ( 1 )
    {
      v74 = (_DWORD *)(*((_QWORD *)a1 + 43) + 8LL * *(int *)(v13 + 24));
      *v74 = a1[48];
      v15 = sub_2E32FD0(v13, *((_QWORD *)a1 + 2));
      if ( v15 )
      {
        v18 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)a1 + 4) + 152LL) + 16LL * *(unsigned int *)(v13 + 24));
        v19 = a1[48];
        if ( v19 + 1 > (unsigned __int64)a1[49] )
        {
          sub_C8D5F0((__int64)(a1 + 46), a1 + 50, v19 + 1, 8u, v16, v17);
          v19 = a1[48];
        }
        *(_QWORD *)(*((_QWORD *)a1 + 23) + 8 * v19) = v18;
        v20 = a1[68];
        v21 = a1[69];
        ++a1[48];
        if ( v20 + 1 > v21 )
        {
          sub_C8D5F0((__int64)(a1 + 66), v75, v20 + 1, 8u, v16, v17);
          v20 = a1[68];
        }
        *(_QWORD *)(*((_QWORD *)a1 + 33) + 8 * v20) = v15;
        ++a1[68];
      }
      if ( *(_BYTE *)(v13 + 216) )
      {
        v22 = *((_QWORD *)a1 + 2);
        v23 = *(__int64 (**)())(*(_QWORD *)v22 + 96LL);
        if ( v23 != sub_2E0FEB0 )
        {
          v66 = ((__int64 (__fastcall *)(__int64, _QWORD))v23)(v22, *(_QWORD *)(v13 + 32));
          if ( v66 )
          {
            v68 = *(_QWORD *)(*(_QWORD *)(*((_QWORD *)a1 + 4) + 152LL) + 16LL * *(unsigned int *)(v13 + 24));
            v69 = a1[48];
            if ( v69 + 1 > (unsigned __int64)a1[49] )
            {
              sub_C8D5F0((__int64)(a1 + 46), a1 + 50, v69 + 1, 8u, v67, v17);
              v69 = a1[48];
            }
            *(_QWORD *)(*((_QWORD *)a1 + 23) + 8 * v69) = v68;
            v70 = a1[68];
            v71 = a1[69];
            ++a1[48];
            if ( v70 + 1 > v71 )
            {
              sub_C8D5F0((__int64)(a1 + 66), v75, v70 + 1, 8u, v67, v17);
              v70 = a1[68];
            }
            *(_QWORD *)(*((_QWORD *)a1 + 33) + 8 * v70) = v66;
            ++a1[68];
          }
        }
      }
      v24 = *(_QWORD *)(v13 + 56);
      if ( v13 + 48 != v24 )
        break;
LABEL_43:
      v45 = sub_2E33000(v13, *((_QWORD *)a1 + 2));
      if ( v45 )
      {
        v46 = *((_QWORD *)a1 + 4);
        v47 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( !v47 )
          BUG();
        v48 = *(_DWORD *)(v47 + 44);
        v49 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        LOBYTE(v50) = v48;
        if ( (*(_QWORD *)v47 & 4) != 0 )
        {
          v52 = *(_QWORD *)(v13 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v48 & 4) != 0 )
          {
            do
              v52 = *(_QWORD *)v52 & 0xFFFFFFFFFFFFFFF8LL;
            while ( (*(_BYTE *)(v52 + 44) & 4) != 0 );
          }
        }
        else
        {
          if ( (v48 & 4) != 0 )
          {
            for ( j = *(_QWORD *)v47; ; j = *(_QWORD *)v49 )
            {
              v49 = j & 0xFFFFFFFFFFFFFFF8LL;
              v50 = *(_DWORD *)(v49 + 44) & 0xFFFFFF;
              if ( (*(_DWORD *)(v49 + 44) & 4) == 0 )
                break;
            }
          }
          v52 = v49;
        }
        if ( (v50 & 8) != 0 )
        {
          do
            v49 = *(_QWORD *)(v49 + 8);
          while ( (*(_BYTE *)(v49 + 44) & 8) != 0 );
        }
        for ( k = *(_QWORD *)(v49 + 8); k != v52; v52 = *(_QWORD *)(v52 + 8) )
        {
          v54 = *(_WORD *)(v52 + 68);
          if ( (unsigned __int16)(v54 - 14) > 4u && v54 != 24 )
            break;
        }
        v55 = *(_DWORD *)(v46 + 144);
        v56 = *(_QWORD *)(v46 + 128);
        if ( v55 )
        {
          v57 = (v55 - 1) & (((unsigned int)v52 >> 9) ^ ((unsigned int)v52 >> 4));
          v58 = (__int64 *)(v56 + 16LL * v57);
          v59 = *v58;
          if ( v52 == *v58 )
          {
LABEL_59:
            v60 = a1[48];
            v61 = v58[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
            if ( v60 + 1 > (unsigned __int64)a1[49] )
            {
              sub_C8D5F0((__int64)(a1 + 46), a1 + 50, v60 + 1, 8u, v56, v44);
              v60 = a1[48];
            }
            *(_QWORD *)(*((_QWORD *)a1 + 23) + 8 * v60) = v61;
            v62 = a1[68];
            v63 = a1[69];
            ++a1[48];
            if ( v62 + 1 > v63 )
            {
              sub_C8D5F0((__int64)(a1 + 66), v75, v62 + 1, 8u, v56, v44);
              v62 = a1[68];
            }
            *(_QWORD *)(*((_QWORD *)a1 + 33) + 8 * v62) = v45;
            ++a1[68];
            goto LABEL_64;
          }
          v65 = 1;
          while ( v59 != -4096 )
          {
            v44 = (unsigned int)(v65 + 1);
            v57 = (v55 - 1) & (v65 + v57);
            v58 = (__int64 *)(v56 + 16LL * v57);
            v59 = *v58;
            if ( *v58 == v52 )
              goto LABEL_59;
            v65 = v44;
          }
        }
        v58 = (__int64 *)(v56 + 16LL * v55);
        goto LABEL_59;
      }
LABEL_64:
      result = a1[48] - *v74;
      v74[1] = result;
      v13 = *(_QWORD *)(v13 + 8);
      if ( v72 == v13 )
        return result;
    }
    v73 = v13;
    v25 = v13 + 48;
    while ( 1 )
    {
      v26 = *(_QWORD *)(v24 + 32);
      v27 = v26 + 40LL * (*(_DWORD *)(v24 + 40) & 0xFFFFFF);
      if ( v26 != v27 )
        break;
LABEL_40:
      if ( (*(_BYTE *)v24 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v24 + 44) & 8) != 0 )
          v24 = *(_QWORD *)(v24 + 8);
      }
      v24 = *(_QWORD *)(v24 + 8);
      if ( v25 == v24 )
      {
        v13 = v73;
        goto LABEL_43;
      }
    }
    while ( 1 )
    {
      while ( *(_BYTE *)v26 != 12 )
      {
        v26 += 40;
        if ( v27 == v26 )
          goto LABEL_40;
      }
      v28 = *((_QWORD *)a1 + 4);
      v29 = v24;
      for ( m = v24; (*(_BYTE *)(m + 44) & 4) != 0; m = *(_QWORD *)m & 0xFFFFFFFFFFFFFFF8LL )
        ;
      if ( (*(_DWORD *)(v24 + 44) & 8) != 0 )
      {
        do
          v29 = *(_QWORD *)(v29 + 8);
        while ( (*(_BYTE *)(v29 + 44) & 8) != 0 );
      }
      for ( n = *(_QWORD *)(v29 + 8); n != m; m = *(_QWORD *)(m + 8) )
      {
        v32 = *(_WORD *)(m + 68);
        if ( (unsigned __int16)(v32 - 14) > 4u && v32 != 24 )
          break;
      }
      v33 = *(unsigned int *)(v28 + 144);
      v34 = *(_QWORD *)(v28 + 128);
      if ( !(_DWORD)v33 )
        goto LABEL_71;
      v35 = (v33 - 1) & (((unsigned int)m >> 9) ^ ((unsigned int)m >> 4));
      v36 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( *v36 != m )
        break;
LABEL_35:
      v38 = v36[1];
      v39 = a1[48];
      v40 = v38 & 0xFFFFFFFFFFFFFFF8LL | 4;
      if ( v39 + 1 > (unsigned __int64)a1[49] )
      {
        v77 = v40;
        sub_C8D5F0((__int64)(a1 + 46), a1 + 50, v39 + 1, 8u, v39 + 1, v17);
        v39 = a1[48];
        v40 = v77;
      }
      *(_QWORD *)(*((_QWORD *)a1 + 23) + 8 * v39) = v40;
      v41 = a1[68];
      v42 = a1[69];
      ++a1[48];
      v43 = *(_QWORD *)(v26 + 24);
      if ( v41 + 1 > v42 )
      {
        v76 = *(_QWORD *)(v26 + 24);
        sub_C8D5F0((__int64)(a1 + 66), v75, v41 + 1, 8u, v43, v17);
        v41 = a1[68];
        v43 = v76;
      }
      v26 += 40;
      *(_QWORD *)(*((_QWORD *)a1 + 33) + 8 * v41) = v43;
      ++a1[68];
      if ( v27 == v26 )
        goto LABEL_40;
    }
    v64 = 1;
    while ( v37 != -4096 )
    {
      v17 = (unsigned int)(v64 + 1);
      v35 = (v33 - 1) & (v64 + v35);
      v36 = (__int64 *)(v34 + 16LL * v35);
      v37 = *v36;
      if ( *v36 == m )
        goto LABEL_35;
      v64 = v17;
    }
LABEL_71:
    v36 = (__int64 *)(v34 + 16 * v33);
    goto LABEL_35;
  }
  return result;
}

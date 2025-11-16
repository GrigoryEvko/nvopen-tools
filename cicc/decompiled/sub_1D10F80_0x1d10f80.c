// Function: sub_1D10F80
// Address: 0x1d10f80
//
__int64 __fastcall sub_1D10F80(_QWORD *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r13
  __int64 result; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // r9d
  unsigned int v14; // ecx
  __int64 *v15; // r13
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // ebx
  __int64 v20; // r9
  __int64 v21; // rsi
  __int64 v22; // r15
  __int64 v23; // r14
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  int v27; // eax
  int v28; // eax
  unsigned int v29; // esi
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 v32; // rdi
  _QWORD *v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r9
  __int64 v36; // rsi
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rax
  int v42; // eax
  int v43; // r9d
  int v44; // r9d
  __int64 v45; // r11
  unsigned int v46; // esi
  int v47; // edx
  _QWORD *v48; // rax
  __int64 v49; // r10
  int v50; // r11d
  int v51; // ecx
  int v52; // r9d
  int v53; // r9d
  __int64 v54; // r11
  unsigned int v55; // ecx
  __int64 v56; // r10
  _QWORD *v57; // rsi
  int v58; // edi
  _QWORD *v59; // rcx
  int v60; // edi
  __int64 v61; // [rsp+0h] [rbp-80h]
  unsigned int v62; // [rsp+0h] [rbp-80h]
  __int64 v63; // [rsp+8h] [rbp-78h]
  int v64; // [rsp+8h] [rbp-78h]
  __int64 v65; // [rsp+18h] [rbp-68h] BYREF
  __int64 v66; // [rsp+20h] [rbp-60h] BYREF
  int v67; // [rsp+28h] [rbp-58h]
  __int64 v68; // [rsp+30h] [rbp-50h]
  __int64 v69; // [rsp+38h] [rbp-48h]
  __int64 v70; // [rsp+40h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 32);
  result = v4 + 16LL * *(unsigned int *)(a2 + 40);
  if ( v4 == result )
    return result;
  while ( (*(_QWORD *)v4 & 6) != 0 )
  {
    v4 += 16;
    if ( result == v4 )
      return result;
  }
  v10 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !*(_QWORD *)(v10 + 256) )
  {
    v28 = sub_1E6B9A0(a1[5], *(_QWORD *)(a2 + 256), byte_3F871B3, 0);
    v29 = *(_DWORD *)(a3 + 24);
    v64 = v28;
    if ( v29 )
    {
      v30 = *(_QWORD *)(a3 + 8);
      v31 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      v32 = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v33 = (_QWORD *)(v30 + 16 * v32);
      v34 = *v33;
      if ( a2 == *v33 )
      {
LABEL_19:
        v35 = a1[77];
        v36 = *(_QWORD *)(a1[2] + 8LL);
        v65 = 0;
        v37 = *(_QWORD *)(v35 + 56);
        v61 = v35;
        v38 = sub_1E0B640(v37, v36 + 960, &v65, 0, v31);
        sub_1DD5BA0(v61 + 16, v38);
        v39 = *(_QWORD *)v38;
        v40 = *a4;
        *(_QWORD *)(v38 + 8) = a4;
        v40 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v38 = v40 | v39 & 7;
        *(_QWORD *)(v40 + 8) = v38;
        v41 = *a4;
        HIDWORD(v66) = 0;
        v68 = 0;
        v69 = 0;
        *a4 = v38 | v41 & 7;
        v70 = 0;
        v67 = v64;
        LODWORD(v66) = 0x10000000;
        sub_1E1A9C0(v38, v37, &v66);
        v42 = *(_DWORD *)(v4 + 8);
        v66 = 0;
        v68 = 0;
        v67 = v42;
        v69 = 0;
        v70 = 0;
        result = sub_1E1A9C0(v38, v37, &v66);
        if ( v65 )
          return sub_161E7C0((__int64)&v65, v65);
        return result;
      }
      v48 = 0;
      v50 = 1;
      while ( v34 != -8 )
      {
        if ( v48 || v34 != -16 )
          v33 = v48;
        LODWORD(v32) = (v29 - 1) & (v50 + v32);
        v34 = *(_QWORD *)(v30 + 16LL * (unsigned int)v32);
        if ( a2 == v34 )
          goto LABEL_19;
        ++v50;
        v48 = v33;
        v33 = (_QWORD *)(v30 + 16LL * (unsigned int)v32);
      }
      v51 = *(_DWORD *)(a3 + 16);
      if ( !v48 )
        v48 = v33;
      ++*(_QWORD *)a3;
      v47 = v51 + 1;
      if ( 4 * (v51 + 1) < 3 * v29 )
      {
        if ( v29 - *(_DWORD *)(a3 + 20) - v47 > v29 >> 3 )
        {
LABEL_24:
          *(_DWORD *)(a3 + 16) = v47;
          if ( *v48 != -8 )
            --*(_DWORD *)(a3 + 20);
          *v48 = a2;
          *((_DWORD *)v48 + 2) = v64;
          goto LABEL_19;
        }
        v62 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
        sub_1D10DC0(a3, v29);
        v52 = *(_DWORD *)(a3 + 24);
        if ( v52 )
        {
          v31 = v62;
          v53 = v52 - 1;
          v54 = *(_QWORD *)(a3 + 8);
          v55 = v53 & v62;
          v47 = *(_DWORD *)(a3 + 16) + 1;
          v48 = (_QWORD *)(v54 + 16LL * (v53 & v62));
          v56 = *v48;
          if ( a2 != *v48 )
          {
            v57 = (_QWORD *)(v54 + 16LL * (v53 & v62));
            v58 = 1;
            v48 = 0;
            while ( v56 != -8 )
            {
              if ( !v48 && v56 == -16 )
                v48 = v57;
              v31 = (unsigned int)(v58 + 1);
              v55 = v53 & (v58 + v55);
              v57 = (_QWORD *)(v54 + 16LL * v55);
              v56 = *v57;
              if ( a2 == *v57 )
              {
                v48 = (_QWORD *)(v54 + 16LL * v55);
                goto LABEL_24;
              }
              ++v58;
            }
            if ( !v48 )
              v48 = v57;
          }
          goto LABEL_24;
        }
LABEL_65:
        ++*(_DWORD *)(a3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a3;
    }
    sub_1D10DC0(a3, 2 * v29);
    v43 = *(_DWORD *)(a3 + 24);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a3 + 8);
      v46 = v44 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v47 = *(_DWORD *)(a3 + 16) + 1;
      v48 = (_QWORD *)(v45 + 16LL * v46);
      v49 = *v48;
      if ( a2 != *v48 )
      {
        v59 = (_QWORD *)(v45 + 16LL * (v44 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
        v60 = 1;
        v48 = 0;
        while ( v49 != -8 )
        {
          if ( v49 == -16 && !v48 )
            v48 = v59;
          v31 = (unsigned int)(v60 + 1);
          v46 = v44 & (v60 + v46);
          v59 = (_QWORD *)(v45 + 16LL * v46);
          v49 = *v59;
          if ( a2 == *v59 )
          {
            v48 = (_QWORD *)(v45 + 16LL * v46);
            goto LABEL_24;
          }
          ++v60;
        }
        if ( !v48 )
          v48 = v59;
      }
      goto LABEL_24;
    }
    goto LABEL_65;
  }
  v11 = *(unsigned int *)(a3 + 24);
  v12 = *(_QWORD *)(a3 + 8);
  if ( !(_DWORD)v11 )
  {
LABEL_14:
    v15 = (__int64 *)(v12 + 16 * v11);
    goto LABEL_6;
  }
  v13 = 1;
  v14 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v15 = (__int64 *)(v12 + 16LL * v14);
  v16 = *v15;
  if ( v10 != *v15 )
  {
    while ( v16 != -8 )
    {
      v14 = (v11 - 1) & (v13 + v14);
      v15 = (__int64 *)(v12 + 16LL * v14);
      v16 = *v15;
      if ( v10 == *v15 )
        goto LABEL_6;
      ++v13;
    }
    goto LABEL_14;
  }
LABEL_6:
  v17 = *(_QWORD *)(a2 + 112);
  v18 = v17 + 16LL * *(unsigned int *)(a2 + 120);
  if ( v17 == v18 )
  {
LABEL_10:
    v19 = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)v17 & 6) == 0 )
      {
        v19 = *(_DWORD *)(v17 + 8);
        if ( v19 )
          break;
      }
      v17 += 16;
      if ( v18 == v17 )
        goto LABEL_10;
    }
  }
  v20 = a1[77];
  v21 = *(_QWORD *)(a1[2] + 8LL);
  v65 = 0;
  v22 = *(_QWORD *)(v20 + 56);
  v63 = v20;
  v23 = sub_1E0B640(v22, v21 + 960, &v65, 0, &v65);
  sub_1DD5BA0(v63 + 16, v23);
  v24 = *(_QWORD *)v23;
  v25 = *a4;
  *(_QWORD *)(v23 + 8) = a4;
  v25 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v23 = v25 | v24 & 7;
  *(_QWORD *)(v25 + 8) = v23;
  v26 = *a4;
  HIDWORD(v66) = 0;
  v68 = 0;
  v67 = v19;
  *a4 = v23 | v26 & 7;
  v69 = 0;
  v70 = 0;
  LODWORD(v66) = 0x10000000;
  sub_1E1A9C0(v23, v22, &v66);
  v27 = *((_DWORD *)v15 + 2);
  v66 = 0;
  v68 = 0;
  v67 = v27;
  v69 = 0;
  v70 = 0;
  result = sub_1E1A9C0(v23, v22, &v66);
  if ( v65 )
    return sub_161E7C0((__int64)&v65, v65);
  return result;
}

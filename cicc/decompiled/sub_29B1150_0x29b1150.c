// Function: sub_29B1150
// Address: 0x29b1150
//
__int64 __fastcall sub_29B1150(__int64 a1, __int64 *a2)
{
  __int64 v2; // r14
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 result; // rax
  int v7; // ecx
  unsigned int v8; // r8d
  int v9; // edi
  __int64 v10; // r12
  int v11; // r13d
  __int64 *v12; // rdx
  int v13; // r11d
  __int64 v14; // r10
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // r12
  char v18; // dh
  __int64 *v19; // rsi
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r11
  __int64 v24; // rcx
  int v25; // eax
  int v26; // edx
  unsigned int v27; // eax
  __int64 *v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rax
  _QWORD *v31; // rdi
  int v32; // edx
  _QWORD *v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rax
  _QWORD *v36; // rax
  _QWORD *v37; // rsi
  __int64 v38; // r13
  __int64 v39; // rbx
  __int64 v40; // r12
  int v41; // edx
  __int64 v42; // rdi
  __int64 v43; // rax
  int v44; // esi
  unsigned int v45; // edx
  __int64 v46; // r8
  unsigned __int64 v47; // rdx
  __int64 v48; // rdi
  void *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rbx
  _QWORD *v53; // r10
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  unsigned int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rdx
  int v61; // r10d
  __int64 v62; // r12
  __int64 v63; // r14
  unsigned int v64; // r13d
  int v65; // edx
  unsigned int v66; // eax
  __int64 v67; // r9
  __int64 v68; // rdx
  int v69; // eax
  int v70; // eax
  unsigned int v71; // ecx
  __int64 v72; // rax
  __int64 v73; // rcx
  __int64 v74; // rcx
  __int64 v75; // rsi
  int v76; // edx
  __int64 v77; // rdi
  __int64 v78; // r15
  int v79; // r8d
  int v80; // r15d
  int v81; // r9d
  int v82; // esi
  int v83; // r8d
  _QWORD *v84; // rax
  int v85; // [rsp+14h] [rbp-9Ch]
  __int64 v86; // [rsp+18h] [rbp-98h]
  __int64 v87; // [rsp+28h] [rbp-88h]
  __int64 v88; // [rsp+30h] [rbp-80h]
  __int64 v89; // [rsp+38h] [rbp-78h]
  _QWORD *v90; // [rsp+38h] [rbp-78h]
  __int64 v91; // [rsp+38h] [rbp-78h]
  __int64 v92; // [rsp+48h] [rbp-68h] BYREF
  void *v93[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v94; // [rsp+70h] [rbp-40h]

  v2 = a1;
  v4 = *a2;
  v5 = *(_QWORD *)(*(_QWORD *)(*a2 + 72) + 80LL);
  if ( v5 && v4 == v5 - 24 )
  {
    v85 = 0;
LABEL_12:
    v17 = *(_QWORD *)v2;
    v94 = 257;
    v19 = (__int64 *)sub_AA4FF0(v4);
    v20 = 0;
    if ( v19 )
      v20 = v18;
    v21 = 1;
    BYTE1(v21) = v20;
    v22 = sub_F36960(*a2, v19, v21, v17, 0, 0, v93, 0);
    v23 = *a2;
    v24 = *(_QWORD *)(v2 + 64);
    v92 = v22;
    v25 = *(_DWORD *)(v2 + 80);
    v88 = v23;
    if ( !v25 )
      goto LABEL_27;
    v26 = v25 - 1;
    v27 = (v25 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v28 = (__int64 *)(v24 + 8LL * v27);
    v29 = *v28;
    if ( v23 != *v28 )
    {
      v82 = 1;
      while ( v29 != -4096 )
      {
        v83 = v82 + 1;
        v27 = v26 & (v82 + v27);
        v28 = (__int64 *)(v24 + 8LL * v27);
        v29 = *v28;
        if ( v23 == *v28 )
          goto LABEL_16;
        v82 = v83;
      }
      goto LABEL_27;
    }
LABEL_16:
    *v28 = -8192;
    v30 = *(unsigned int *)(v2 + 96);
    v31 = *(_QWORD **)(v2 + 88);
    --*(_DWORD *)(v2 + 72);
    v32 = v30;
    v30 *= 8;
    ++*(_DWORD *)(v2 + 76);
    v33 = (_QWORD *)((char *)v31 + v30);
    v34 = v30 >> 3;
    v35 = v30 >> 5;
    if ( v35 )
    {
      v36 = &v31[4 * v35];
      while ( v23 != *v31 )
      {
        if ( v23 == v31[1] )
        {
          v37 = ++v31 + 1;
          goto LABEL_24;
        }
        if ( v23 == v31[2] )
        {
          v31 += 2;
          v37 = v31 + 1;
          goto LABEL_24;
        }
        if ( v23 == v31[3] )
        {
          v31 += 3;
          v37 = v31 + 1;
          goto LABEL_24;
        }
        v31 += 4;
        if ( v36 == v31 )
        {
          v34 = v33 - v31;
          goto LABEL_88;
        }
      }
      goto LABEL_23;
    }
LABEL_88:
    switch ( v34 )
    {
      case 2LL:
        v84 = v31;
        break;
      case 3LL:
        v37 = v31 + 1;
        v84 = v31 + 1;
        if ( v23 == *v31 )
          goto LABEL_24;
        break;
      case 1LL:
        goto LABEL_95;
      default:
LABEL_91:
        v31 = v33;
        v37 = v33 + 1;
        goto LABEL_24;
    }
    v31 = v84 + 1;
    if ( v23 == *v84 )
    {
      v31 = v84;
      goto LABEL_23;
    }
LABEL_95:
    if ( v23 != *v31 )
      goto LABEL_91;
LABEL_23:
    v37 = v31 + 1;
LABEL_24:
    if ( v37 != v33 )
    {
      memmove(v31, v37, (char *)v33 - (char *)v37);
      v32 = *(_DWORD *)(v2 + 96);
    }
    *(_DWORD *)(v2 + 96) = v32 - 1;
LABEL_27:
    sub_29B0C40(v2 + 56, &v92);
    result = v92;
    *a2 = v92;
    if ( !v85 )
      return result;
    v38 = *(_QWORD *)(v88 + 56);
    if ( !v38 )
      BUG();
    result = *(_DWORD *)(v38 - 20) & 0x7FFFFFF;
    if ( (*(_DWORD *)(v38 - 20) & 0x7FFFFFF) != 0 )
    {
      v39 = 8LL * (unsigned int)result;
      v40 = 0;
      while ( 1 )
      {
        v41 = *(_DWORD *)(v2 + 80);
        v42 = *(_QWORD *)(v2 + 64);
        v43 = *(_QWORD *)(*(_QWORD *)(v38 - 32) + 32LL * *(unsigned int *)(v38 + 48) + v40);
        if ( v41 )
        {
          v44 = v41 - 1;
          v45 = (v41 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v46 = *(_QWORD *)(v42 + 8LL * v45);
          if ( v43 == v46 )
          {
LABEL_33:
            v47 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
            if ( v47 == v43 + 48 )
            {
              v48 = 0;
            }
            else
            {
              if ( !v47 )
                BUG();
              v48 = v47 - 24;
              if ( (unsigned int)*(unsigned __int8 *)(v47 - 24) - 30 >= 0xB )
                v48 = 0;
            }
            sub_BD2ED0(v48, v88, v92);
          }
          else
          {
            v81 = 1;
            while ( v46 != -4096 )
            {
              v45 = v44 & (v81 + v45);
              v46 = *(_QWORD *)(v42 + 8LL * v45);
              if ( v43 == v46 )
                goto LABEL_33;
              ++v81;
            }
          }
        }
        v40 += 8;
        if ( v39 == v40 )
        {
          result = v88;
          v38 = *(_QWORD *)(v88 + 56);
          goto LABEL_40;
        }
      }
    }
    while ( 1 )
    {
LABEL_40:
      if ( !v38 )
        BUG();
      if ( *(_BYTE *)(v38 - 24) != 84 )
        return result;
      v93[0] = (void *)sub_BD5D20(v38 - 24);
      v93[2] = ".ce";
      v94 = 773;
      v93[1] = v49;
      v89 = *(_QWORD *)(v38 - 16);
      v50 = sub_BD2DA0(80);
      v51 = v89;
      v52 = v50;
      if ( v50 )
      {
        v90 = (_QWORD *)v50;
        sub_B44260(v50, v51, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v52 + 72) = v85 + 1;
        sub_BD6B50((unsigned __int8 *)v52, (const char **)v93);
        sub_BD2A10(v52, *(_DWORD *)(v52 + 72), 1);
        v53 = v90;
      }
      else
      {
        v53 = 0;
      }
      v54 = v87;
      LOWORD(v54) = 1;
      v87 = v54;
      sub_B44220(v53, *(_QWORD *)(v92 + 56), v54);
      sub_BD84D0(v38 - 24, v52);
      v55 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
      if ( v55 == *(_DWORD *)(v52 + 72) )
      {
        sub_B48D90(v52);
        v55 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
      }
      v56 = (v55 + 1) & 0x7FFFFFF;
      v57 = v56 | *(_DWORD *)(v52 + 4) & 0xF8000000;
      v58 = *(_QWORD *)(v52 - 8) + 32LL * (unsigned int)(v56 - 1);
      *(_DWORD *)(v52 + 4) = v57;
      if ( *(_QWORD *)v58 )
      {
        v59 = *(_QWORD *)(v58 + 8);
        **(_QWORD **)(v58 + 16) = v59;
        if ( v59 )
          *(_QWORD *)(v59 + 16) = *(_QWORD *)(v58 + 16);
      }
      *(_QWORD *)v58 = v38 - 24;
      v60 = *(_QWORD *)(v38 - 8);
      *(_QWORD *)(v58 + 8) = v60;
      if ( v60 )
        *(_QWORD *)(v60 + 16) = v58 + 8;
      *(_QWORD *)(v58 + 16) = v38 - 8;
      *(_QWORD *)(v38 - 8) = v58;
      result = *(_QWORD *)(v52 - 8)
             + 32LL * *(unsigned int *)(v52 + 72)
             + 8LL * ((*(_DWORD *)(v52 + 4) & 0x7FFFFFFu) - 1);
      *(_QWORD *)result = v88;
      v61 = *(_DWORD *)(v38 - 20);
      if ( (v61 & 0x7FFFFFF) != 0 )
        break;
LABEL_67:
      v38 = *(_QWORD *)(v38 + 8);
    }
    v91 = v38 - 24;
    v62 = v2;
    v63 = v38;
    v64 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v75 = *(_QWORD *)(v63 - 32);
        v76 = *(_DWORD *)(v62 + 80);
        v77 = *(_QWORD *)(v62 + 64);
        v78 = *(_QWORD *)(v75 + 32LL * *(unsigned int *)(v63 + 48) + 8LL * v64);
        if ( v76 )
          break;
LABEL_65:
        ++v64;
        result = v61 & 0x7FFFFFF;
        if ( (_DWORD)result == v64 )
          goto LABEL_66;
      }
      v65 = v76 - 1;
      v66 = v65 & (((unsigned int)v78 >> 9) ^ ((unsigned int)v78 >> 4));
      v67 = *(_QWORD *)(v77 + 8LL * v66);
      if ( v78 != v67 )
      {
        v79 = 1;
        while ( v67 != -4096 )
        {
          v66 = v65 & (v79 + v66);
          v67 = *(_QWORD *)(v77 + 8LL * v66);
          if ( v78 == v67 )
            goto LABEL_54;
          ++v79;
        }
        goto LABEL_65;
      }
LABEL_54:
      v68 = *(_QWORD *)(v75 + 32LL * v64);
      v69 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
      if ( v69 == *(_DWORD *)(v52 + 72) )
      {
        v86 = *(_QWORD *)(v75 + 32LL * v64);
        sub_B48D90(v52);
        v68 = v86;
        v69 = *(_DWORD *)(v52 + 4) & 0x7FFFFFF;
      }
      v70 = (v69 + 1) & 0x7FFFFFF;
      v71 = v70 | *(_DWORD *)(v52 + 4) & 0xF8000000;
      v72 = *(_QWORD *)(v52 - 8) + 32LL * (unsigned int)(v70 - 1);
      *(_DWORD *)(v52 + 4) = v71;
      if ( *(_QWORD *)v72 )
      {
        v73 = *(_QWORD *)(v72 + 8);
        **(_QWORD **)(v72 + 16) = v73;
        if ( v73 )
          *(_QWORD *)(v73 + 16) = *(_QWORD *)(v72 + 16);
      }
      *(_QWORD *)v72 = v68;
      if ( v68 )
      {
        v74 = *(_QWORD *)(v68 + 16);
        *(_QWORD *)(v72 + 8) = v74;
        if ( v74 )
          *(_QWORD *)(v74 + 16) = v72 + 8;
        *(_QWORD *)(v72 + 16) = v68 + 16;
        *(_QWORD *)(v68 + 16) = v72;
      }
      *(_QWORD *)(*(_QWORD *)(v52 - 8)
                + 32LL * *(unsigned int *)(v52 + 72)
                + 8LL * ((*(_DWORD *)(v52 + 4) & 0x7FFFFFFu) - 1)) = v78;
      sub_B48BF0(v91, v64, 1);
      v61 = *(_DWORD *)(v63 - 20);
      result = v61 & 0x7FFFFFF;
      if ( (_DWORD)result == v64 )
      {
LABEL_66:
        v38 = v63;
        v2 = v62;
        goto LABEL_67;
      }
    }
  }
  result = *(_QWORD *)(v4 + 56);
  if ( !result )
    BUG();
  if ( *(_BYTE *)(result - 24) != 84 )
    return result;
  v7 = *(_DWORD *)(result - 20) & 0x7FFFFFF;
  if ( !v7 )
    return result;
  v8 = 0;
  v9 = *(_DWORD *)(a1 + 80);
  v10 = *(_QWORD *)(v2 + 64);
  v11 = 0;
  v12 = (__int64 *)(*(_QWORD *)(result - 32) + 32LL * *(unsigned int *)(result + 48));
  result = (__int64)(v12 + 1);
  v13 = v9 - 1;
  v14 = (__int64)&v12[(unsigned int)(v7 - 1) + 1];
  do
  {
    while ( 1 )
    {
      v16 = *v12;
      if ( !v9 )
        goto LABEL_10;
      result = v13 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v15 = *(_QWORD *)(v10 + 8 * result);
      if ( v15 != v16 )
        break;
LABEL_8:
      ++v12;
      ++v11;
      if ( (__int64 *)v14 == v12 )
        goto LABEL_11;
    }
    v80 = 1;
    while ( v15 != -4096 )
    {
      result = v13 & (unsigned int)(v80 + result);
      v15 = *(_QWORD *)(v10 + 8LL * (unsigned int)result);
      if ( v16 == v15 )
        goto LABEL_8;
      ++v80;
    }
LABEL_10:
    ++v12;
    ++v8;
  }
  while ( (__int64 *)v14 != v12 );
LABEL_11:
  v85 = v11;
  if ( v8 > 1 )
    goto LABEL_12;
  return result;
}

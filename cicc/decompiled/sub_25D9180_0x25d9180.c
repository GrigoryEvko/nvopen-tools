// Function: sub_25D9180
// Address: 0x25d9180
//
__int64 __fastcall sub_25D9180(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r9
  unsigned int v8; // esi
  __int64 v9; // rdi
  int v10; // r11d
  _QWORD *v11; // r10
  unsigned int v12; // r13d
  __int64 v13; // rcx
  _QWORD *v14; // rdx
  __int64 v15; // rax
  __int64 *v16; // rax
  __int64 v17; // r13
  char v18; // cl
  char v19; // r12
  __int64 result; // rax
  __int64 *i; // rax
  __int64 v22; // r14
  unsigned __int64 v23; // rsi
  unsigned __int8 *v24; // rdi
  unsigned __int8 *v25; // r8
  unsigned int v26; // esi
  __int64 v27; // r9
  int v28; // r11d
  unsigned int v29; // edi
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rdi
  int v34; // eax
  int v35; // eax
  int v36; // ecx
  int v37; // edx
  __int64 v38; // rsi
  __int64 v39; // rdx
  int v40; // ecx
  __int64 v41; // rcx
  __int64 v42; // rdx
  int v43; // ecx
  int v44; // ecx
  __int64 v45; // r10
  __int64 v46; // rsi
  __int64 v47; // rdi
  int v48; // r11d
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // r10
  int v52; // r11d
  __int64 v53; // rsi
  __int64 v54; // rdi
  int v55; // r8d
  int v56; // r8d
  __int64 v57; // r10
  unsigned int v58; // ecx
  __int64 v59; // r9
  int v60; // edi
  _QWORD *v61; // rsi
  int v62; // edi
  int v63; // edi
  __int64 v64; // r8
  int v65; // ecx
  _QWORD *v66; // r9
  unsigned int v67; // r13d
  __int64 v68; // rsi
  unsigned __int8 *v69; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v70; // [rsp+8h] [rbp-58h]
  __int64 v71; // [rsp+10h] [rbp-50h]
  unsigned int v72; // [rsp+1Ch] [rbp-44h]
  __int64 v73; // [rsp+20h] [rbp-40h]

  v4 = a1 + 440;
  v8 = *(_DWORD *)(a1 + 464);
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 440);
    goto LABEL_83;
  }
  v9 = *(_QWORD *)(a1 + 448);
  v10 = 1;
  v11 = 0;
  v12 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  LODWORD(v13) = (v8 - 1) & v12;
  v14 = (_QWORD *)(v9 + 136LL * (unsigned int)v13);
  v15 = *v14;
  if ( *v14 != a3 )
  {
    while ( v15 != -4096 )
    {
      if ( v15 == -8192 && !v11 )
        v11 = v14;
      v13 = (v8 - 1) & ((_DWORD)v13 + v10);
      v14 = (_QWORD *)(v9 + 136 * v13);
      v15 = *v14;
      if ( *v14 == a3 )
        goto LABEL_3;
      ++v10;
    }
    v34 = *(_DWORD *)(a1 + 456);
    if ( v11 )
      v14 = v11;
    ++*(_QWORD *)(a1 + 440);
    v35 = v34 + 1;
    if ( 4 * v35 < 3 * v8 )
    {
      if ( v8 - *(_DWORD *)(a1 + 460) - v35 > v8 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(a1 + 456) = v35;
        if ( *v14 != -4096 )
          --*(_DWORD *)(a1 + 460);
        *v14 = a3;
        memset(v14 + 1, 0, 0x80u);
        v14[1] = v14 + 3;
        v14[2] = 0x400000000LL;
        v14[14] = v14 + 12;
        v14[15] = v14 + 12;
        v16 = v14 + 1;
        goto LABEL_31;
      }
      sub_25D8AF0(v4, v8);
      v62 = *(_DWORD *)(a1 + 464);
      if ( v62 )
      {
        v63 = v62 - 1;
        v64 = *(_QWORD *)(a1 + 448);
        v65 = 1;
        v66 = 0;
        v67 = v63 & v12;
        v14 = (_QWORD *)(v64 + 136LL * v67);
        v68 = *v14;
        v35 = *(_DWORD *)(a1 + 456) + 1;
        if ( *v14 != a3 )
        {
          while ( v68 != -4096 )
          {
            if ( !v66 && v68 == -8192 )
              v66 = v14;
            v67 = v63 & (v65 + v67);
            v14 = (_QWORD *)(v64 + 136LL * v67);
            v68 = *v14;
            if ( *v14 == a3 )
              goto LABEL_28;
            ++v65;
          }
          if ( v66 )
            v14 = v66;
        }
        goto LABEL_28;
      }
LABEL_112:
      ++*(_DWORD *)(a1 + 456);
      BUG();
    }
LABEL_83:
    sub_25D8AF0(v4, 2 * v8);
    v55 = *(_DWORD *)(a1 + 464);
    if ( v55 )
    {
      v56 = v55 - 1;
      v57 = *(_QWORD *)(a1 + 448);
      v58 = v56 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v14 = (_QWORD *)(v57 + 136LL * v58);
      v59 = *v14;
      v35 = *(_DWORD *)(a1 + 456) + 1;
      if ( *v14 != a3 )
      {
        v60 = 1;
        v61 = 0;
        while ( v59 != -4096 )
        {
          if ( !v61 && v59 == -8192 )
            v61 = v14;
          v58 = v56 & (v60 + v58);
          v14 = (_QWORD *)(v57 + 136LL * v58);
          v59 = *v14;
          if ( *v14 == a3 )
            goto LABEL_28;
          ++v60;
        }
        if ( v61 )
          v14 = v61;
      }
      goto LABEL_28;
    }
    goto LABEL_112;
  }
LABEL_3:
  v16 = v14 + 1;
  if ( v14[16] )
  {
    v17 = v14[14];
    v18 = 0;
    v73 = (__int64)(v14 + 12);
    goto LABEL_5;
  }
LABEL_31:
  v17 = *v16;
  v18 = 1;
  v73 = *v16 + 16LL * *((unsigned int *)v16 + 2);
LABEL_5:
  v19 = v18;
  result = a1 + 296;
  v72 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v71 = a1 + 296;
  if ( v18 )
    goto LABEL_16;
  if ( v73 != v17 )
  {
    while ( 1 )
    {
      for ( i = (__int64 *)(v17 + 32); ; i = (__int64 *)v17 )
      {
        v22 = *i;
        v23 = i[1] + a4;
        v24 = (unsigned __int8 *)sub_E027A0(*(_QWORD *)(*i - 32), v23, *(_QWORD *)(a2 + 40), *i);
        if ( v24 )
          break;
        if ( !*(_BYTE *)(a1 + 500) )
          goto LABEL_64;
        v38 = *(_QWORD *)(a1 + 480);
        v42 = v38 + 8LL * *(unsigned int *)(a1 + 492);
        v40 = *(_DWORD *)(a1 + 492);
        result = v38;
        if ( v38 != v42 )
        {
          while ( v22 != *(_QWORD *)result )
          {
            result += 8;
            if ( v42 == result )
              goto LABEL_14;
          }
LABEL_57:
          v41 = (unsigned int)(v40 - 1);
          *(_DWORD *)(a1 + 492) = v41;
          *(_QWORD *)result = *(_QWORD *)(v38 + 8 * v41);
          ++*(_QWORD *)(a1 + 472);
        }
LABEL_14:
        if ( !v19 )
          goto LABEL_50;
LABEL_15:
        v17 += 16;
LABEL_16:
        if ( v73 == v17 )
          return result;
      }
      v25 = sub_BD3990(v24, v23);
      if ( *v25 )
      {
        if ( *(_BYTE *)(a1 + 500) )
        {
          result = *(unsigned int *)(a1 + 492);
          v38 = *(_QWORD *)(a1 + 480);
          v39 = v38 + 8 * result;
          v40 = *(_DWORD *)(a1 + 492);
          if ( v38 == v39 )
            goto LABEL_14;
          result = *(_QWORD *)(a1 + 480);
          while ( v22 != *(_QWORD *)result )
          {
            result += 8;
            if ( v39 == result )
              goto LABEL_14;
          }
          goto LABEL_57;
        }
LABEL_64:
        result = (__int64)sub_C8CA60(a1 + 472, v22);
        if ( result )
        {
          *(_QWORD *)result = -2;
          ++*(_DWORD *)(a1 + 496);
          ++*(_QWORD *)(a1 + 472);
        }
        goto LABEL_14;
      }
      v26 = *(_DWORD *)(a1 + 320);
      if ( !v26 )
        break;
      v27 = *(_QWORD *)(a1 + 304);
      v28 = 1;
      v29 = (v26 - 1) & v72;
      v30 = v27 + 72LL * v29;
      v31 = 0;
      v32 = *(_QWORD *)v30;
      if ( a2 != *(_QWORD *)v30 )
      {
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v31 )
            v31 = v30;
          v29 = (v26 - 1) & (v28 + v29);
          v30 = v27 + 72LL * v29;
          v32 = *(_QWORD *)v30;
          if ( a2 == *(_QWORD *)v30 )
            goto LABEL_12;
          ++v28;
        }
        v36 = *(_DWORD *)(a1 + 312);
        if ( !v31 )
          v31 = v30;
        ++*(_QWORD *)(a1 + 296);
        v37 = v36 + 1;
        if ( 4 * (v36 + 1) < 3 * v26 )
        {
          if ( v26 - *(_DWORD *)(a1 + 316) - v37 <= v26 >> 3 )
          {
            v70 = v25;
            sub_25D83A0(v71, v26);
            v49 = *(_DWORD *)(a1 + 320);
            if ( !v49 )
            {
LABEL_111:
              ++*(_DWORD *)(a1 + 312);
              BUG();
            }
            v50 = v49 - 1;
            v51 = *(_QWORD *)(a1 + 304);
            v27 = 0;
            v25 = v70;
            v52 = 1;
            LODWORD(v53) = v50 & v72;
            v37 = *(_DWORD *)(a1 + 312) + 1;
            v31 = v51 + 72LL * (v50 & v72);
            v54 = *(_QWORD *)v31;
            if ( a2 != *(_QWORD *)v31 )
            {
              while ( v54 != -4096 )
              {
                if ( v54 == -8192 && !v27 )
                  v27 = v31;
                v53 = v50 & (unsigned int)(v53 + v52);
                v31 = v51 + 72 * v53;
                v54 = *(_QWORD *)v31;
                if ( a2 == *(_QWORD *)v31 )
                  goto LABEL_42;
                ++v52;
              }
              goto LABEL_79;
            }
          }
          goto LABEL_42;
        }
LABEL_67:
        v69 = v25;
        sub_25D83A0(v71, 2 * v26);
        v43 = *(_DWORD *)(a1 + 320);
        if ( !v43 )
          goto LABEL_111;
        v44 = v43 - 1;
        v45 = *(_QWORD *)(a1 + 304);
        v25 = v69;
        LODWORD(v46) = v44 & v72;
        v37 = *(_DWORD *)(a1 + 312) + 1;
        v31 = v45 + 72LL * (v44 & v72);
        v47 = *(_QWORD *)v31;
        if ( a2 != *(_QWORD *)v31 )
        {
          v48 = 1;
          v27 = 0;
          while ( v47 != -4096 )
          {
            if ( !v27 && v47 == -8192 )
              v27 = v31;
            v46 = v44 & (unsigned int)(v46 + v48);
            v31 = v45 + 72 * v46;
            v47 = *(_QWORD *)v31;
            if ( a2 == *(_QWORD *)v31 )
              goto LABEL_42;
            ++v48;
          }
LABEL_79:
          if ( v27 )
            v31 = v27;
        }
LABEL_42:
        *(_DWORD *)(a1 + 312) = v37;
        if ( *(_QWORD *)v31 != -4096 )
          --*(_DWORD *)(a1 + 316);
        *(_QWORD *)v31 = a2;
        v33 = v31 + 8;
        *(_QWORD *)(v31 + 8) = 0;
        *(_QWORD *)(v31 + 16) = v31 + 40;
        *(_QWORD *)(v31 + 24) = 4;
        *(_DWORD *)(v31 + 32) = 0;
        *(_BYTE *)(v31 + 36) = 1;
        goto LABEL_45;
      }
LABEL_12:
      v33 = v30 + 8;
      if ( !*(_BYTE *)(v30 + 36) )
        goto LABEL_13;
LABEL_45:
      result = *(_QWORD *)(v33 + 8);
      v32 = *(unsigned int *)(v33 + 20);
      v30 = result + 8 * v32;
      if ( result != v30 )
      {
        while ( v25 != *(unsigned __int8 **)result )
        {
          result += 8;
          if ( v30 == result )
            goto LABEL_48;
        }
        goto LABEL_14;
      }
LABEL_48:
      if ( (unsigned int)v32 >= *(_DWORD *)(v33 + 16) )
      {
LABEL_13:
        result = (__int64)sub_C8CC70(v33, (__int64)v25, v30, v32, (__int64)v25, v27);
        goto LABEL_14;
      }
      *(_DWORD *)(v33 + 20) = v32 + 1;
      *(_QWORD *)v30 = v25;
      ++*(_QWORD *)v33;
      if ( v19 )
        goto LABEL_15;
LABEL_50:
      result = sub_220EF30(v17);
      v17 = result;
      if ( v73 == result )
        return result;
    }
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_67;
  }
  return result;
}

// Function: sub_1F63C50
// Address: 0x1f63c50
//
__int64 __fastcall sub_1F63C50(__int64 a1, __int64 a2, unsigned __int32 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // r8
  bool v19; // zf
  __int64 v20; // rax
  __m128i *v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // r15d
  unsigned int v24; // esi
  __int64 v25; // r9
  unsigned int v26; // edi
  __int64 result; // rax
  __int64 v28; // rdx
  __int64 v29; // r13
  __int64 v30; // r15
  _QWORD *v31; // r13
  _QWORD *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdi
  int v40; // eax
  int v41; // edx
  __int64 v42; // rsi
  __int64 v43; // rcx
  int v44; // edi
  __int64 v45; // rax
  __m128i *v46; // rax
  __int64 v47; // rdx
  unsigned int v48; // r13d
  unsigned int v49; // esi
  __int64 v50; // rdi
  unsigned int v51; // ecx
  __int64 v52; // rdx
  __int64 v53; // r14
  __int64 i; // rbx
  unsigned int v55; // ecx
  __int64 v56; // rax
  int v57; // r10d
  int v58; // r10d
  __int64 v59; // r11
  __int64 v60; // rdx
  int v61; // ecx
  __int64 v62; // r9
  int v63; // r11d
  __int64 v64; // r13
  int v65; // edi
  int v66; // edx
  int v67; // ecx
  int v68; // ecx
  __int64 v69; // r11
  unsigned int v70; // esi
  __int64 v71; // rdi
  int v72; // r10d
  __int64 v73; // r9
  int v74; // esi
  int v75; // esi
  __int64 v76; // r11
  int v77; // r10d
  unsigned int v78; // ecx
  __int64 v79; // rdi
  int v80; // r15d
  __int64 v81; // r10
  int v82; // ecx
  int v83; // edi
  int v84; // edi
  __int64 v85; // r9
  __int64 v86; // r10
  __int64 v87; // r14
  int v88; // edx
  __int64 v89; // rsi
  int v90; // edi
  __int64 v91; // rsi
  unsigned int v92; // [rsp+Ch] [rbp-64h]
  __int64 v93; // [rsp+10h] [rbp-60h]
  __int64 v94; // [rsp+10h] [rbp-60h]
  __int64 v95; // [rsp+10h] [rbp-60h]
  __int64 v97; // [rsp+18h] [rbp-58h]
  __int64 v98; // [rsp+18h] [rbp-58h]
  __int64 v99; // [rsp+18h] [rbp-58h]
  __m128i v100; // [rsp+20h] [rbp-50h] BYREF
  __int64 v101; // [rsp+30h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 40);
  if ( *(_BYTE *)(a2 + 16) != 34 )
  {
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      result = (v40 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v43 = *(_QWORD *)(v42 + 16 * result);
      if ( a2 == v43 )
        return result;
      v44 = 1;
      while ( v43 != -8 )
      {
        a6 = v44 + 1;
        result = v41 & (unsigned int)(v44 + result);
        v43 = *(_QWORD *)(v42 + 16LL * (unsigned int)result);
        if ( a2 == v43 )
          return result;
        ++v44;
      }
    }
    v100.m128i_i64[1] = 0;
    v100.m128i_i8[4] = 1;
    v100.m128i_i32[0] = a3;
    v45 = *(unsigned int *)(a1 + 488);
    v101 = v8;
    if ( (unsigned int)v45 >= *(_DWORD *)(a1 + 492) )
    {
      v98 = v8;
      sub_16CD150(a1 + 480, (const void *)(a1 + 496), 0, 24, v8, a6);
      v45 = *(unsigned int *)(a1 + 488);
      v8 = v98;
    }
    v46 = (__m128i *)(*(_QWORD *)(a1 + 480) + 24 * v45);
    v47 = v101;
    *v46 = _mm_loadu_si128(&v100);
    v46[1].m128i_i64[0] = v47;
    v48 = *(_DWORD *)(a1 + 488);
    v49 = *(_DWORD *)(a1 + 24);
    *(_DWORD *)(a1 + 488) = v48 + 1;
    if ( v49 )
    {
      v50 = *(_QWORD *)(a1 + 8);
      v51 = (v49 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      result = v50 + 16LL * v51;
      v52 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_58;
      v80 = 1;
      v81 = 0;
      while ( v52 != -8 )
      {
        if ( v81 || v52 != -16 )
          result = v81;
        v51 = (v49 - 1) & (v80 + v51);
        v52 = *(_QWORD *)(v50 + 16LL * v51);
        if ( a2 == v52 )
        {
          result = v50 + 16LL * v51;
          goto LABEL_58;
        }
        ++v80;
        v81 = result;
        result = v50 + 16LL * v51;
      }
      v82 = *(_DWORD *)(a1 + 16);
      if ( v81 )
        result = v81;
      ++*(_QWORD *)a1;
      v61 = v82 + 1;
      if ( 4 * v61 < 3 * v49 )
      {
        if ( v49 - *(_DWORD *)(a1 + 20) - v61 <= v49 >> 3 )
        {
          v99 = v8;
          sub_1F61920(a1, v49);
          v83 = *(_DWORD *)(a1 + 24);
          if ( !v83 )
            goto LABEL_146;
          v84 = v83 - 1;
          v85 = *(_QWORD *)(a1 + 8);
          v86 = 0;
          LODWORD(v87) = v84 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v8 = v99;
          v88 = 1;
          v61 = *(_DWORD *)(a1 + 16) + 1;
          result = v85 + 16LL * (unsigned int)v87;
          v89 = *(_QWORD *)result;
          if ( a2 != *(_QWORD *)result )
          {
            while ( v89 != -8 )
            {
              if ( !v86 && v89 == -16 )
                v86 = result;
              v87 = v84 & (unsigned int)(v87 + v88);
              result = v85 + 16 * v87;
              v89 = *(_QWORD *)result;
              if ( a2 == *(_QWORD *)result )
                goto LABEL_75;
              ++v88;
            }
            if ( v86 )
              result = v86;
          }
        }
LABEL_75:
        *(_DWORD *)(a1 + 16) = v61;
        if ( *(_QWORD *)result != -8 )
          --*(_DWORD *)(a1 + 20);
        *(_QWORD *)result = a2;
        *(_DWORD *)(result + 8) = 0;
LABEL_58:
        *(_DWORD *)(result + 8) = v48;
        v53 = *(_QWORD *)(v8 + 8);
        if ( v53 )
        {
          while ( 1 )
          {
            result = (__int64)sub_1648700(v53);
            if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
              break;
            v53 = *(_QWORD *)(v53 + 8);
            if ( !v53 )
              goto LABEL_61;
          }
LABEL_68:
          result = sub_1F60340(*(_QWORD *)(result + 40), *(_QWORD *)(a2 - 24));
          if ( result )
          {
            v56 = sub_157ED20(result);
            result = sub_1F63C50(a1, v56, v48);
          }
          while ( 1 )
          {
            v53 = *(_QWORD *)(v53 + 8);
            if ( !v53 )
              break;
            result = (__int64)sub_1648700(v53);
            if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
              goto LABEL_68;
          }
        }
LABEL_61:
        for ( i = *(_QWORD *)(a2 + 8); i; i = *(_QWORD *)(i + 8) )
        {
          result = (__int64)sub_1648700(i);
          v55 = *(unsigned __int8 *)(result + 16) - 34;
          if ( v55 <= 0x36 )
          {
            result = 1LL << v55;
            if ( ((1LL << v55) & 0x40018000000001LL) != 0 )
              sub_16BD130("Cleanup funclets for the SEH personality cannot contain exceptional actions", 1u);
          }
        }
        return result;
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    v97 = v8;
    sub_1F61920(a1, 2 * v49);
    v57 = *(_DWORD *)(a1 + 24);
    if ( !v57 )
      goto LABEL_146;
    v58 = v57 - 1;
    v59 = *(_QWORD *)(a1 + 8);
    v8 = v97;
    LODWORD(v60) = v58 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v61 = *(_DWORD *)(a1 + 16) + 1;
    result = v59 + 16LL * (unsigned int)v60;
    v62 = *(_QWORD *)result;
    if ( a2 != *(_QWORD *)result )
    {
      v90 = 1;
      v91 = 0;
      while ( v62 != -8 )
      {
        if ( !v91 && v62 == -16 )
          v91 = result;
        v60 = v58 & (unsigned int)(v60 + v90);
        result = v59 + 16 * v60;
        v62 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_75;
        ++v90;
      }
      if ( v91 )
        result = v91;
    }
    goto LABEL_75;
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a2 - 8);
  else
    v9 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  v10 = (__int64 *)(v9 + 24);
  v11 = (__int64 *)(v9 + 48);
  v93 = *(_QWORD *)(a2 + 40);
  if ( (*(_BYTE *)(a2 + 18) & 1) == 0 )
    v11 = v10;
  v12 = sub_1523720(*v11);
  v13 = sub_157ED20(v12);
  v14 = *(_QWORD *)(v13 + 40);
  v15 = v13;
  v16 = sub_1649C60(*(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF)));
  v18 = v93;
  v100.m128i_i8[4] = 0;
  v19 = *(_BYTE *)(v16 + 16) == 0;
  v101 = v14;
  if ( !v19 )
    v16 = 0;
  v100.m128i_i32[0] = a3;
  v100.m128i_i64[1] = v16;
  v20 = *(unsigned int *)(a1 + 488);
  if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 492) )
  {
    sub_16CD150(a1 + 480, (const void *)(a1 + 496), 0, 24, v93, v17);
    v20 = *(unsigned int *)(a1 + 488);
    v18 = v93;
  }
  v21 = (__m128i *)(*(_QWORD *)(a1 + 480) + 24 * v20);
  v22 = v101;
  *v21 = _mm_loadu_si128(&v100);
  v21[1].m128i_i64[0] = v22;
  v23 = *(_DWORD *)(a1 + 488);
  v24 = *(_DWORD *)(a1 + 24);
  *(_DWORD *)(a1 + 488) = v23 + 1;
  if ( !v24 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_88;
  }
  v25 = *(_QWORD *)(a1 + 8);
  v26 = (v24 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  result = v25 + 16LL * v26;
  v28 = *(_QWORD *)result;
  if ( a2 == *(_QWORD *)result )
    goto LABEL_12;
  v63 = 1;
  v64 = 0;
  while ( v28 != -8 )
  {
    if ( !v64 && v28 == -16 )
      v64 = result;
    v26 = (v24 - 1) & (v63 + v26);
    result = v25 + 16LL * v26;
    v28 = *(_QWORD *)result;
    if ( a2 == *(_QWORD *)result )
      goto LABEL_12;
    ++v63;
  }
  v65 = *(_DWORD *)(a1 + 16);
  if ( v64 )
    result = v64;
  ++*(_QWORD *)a1;
  v66 = v65 + 1;
  if ( 4 * (v65 + 1) >= 3 * v24 )
  {
LABEL_88:
    v94 = v18;
    sub_1F61920(a1, 2 * v24);
    v67 = *(_DWORD *)(a1 + 24);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 8);
      v18 = v94;
      v70 = v68 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v66 = *(_DWORD *)(a1 + 16) + 1;
      result = v69 + 16LL * v70;
      v71 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_84;
      v72 = 1;
      v73 = 0;
      while ( v71 != -8 )
      {
        if ( !v73 && v71 == -16 )
          v73 = result;
        v70 = v68 & (v72 + v70);
        result = v69 + 16LL * v70;
        v71 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_84;
        ++v72;
      }
LABEL_92:
      if ( v73 )
        result = v73;
      goto LABEL_84;
    }
LABEL_146:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
  if ( v24 - *(_DWORD *)(a1 + 20) - v66 <= v24 >> 3 )
  {
    v92 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v95 = v18;
    sub_1F61920(a1, v24);
    v74 = *(_DWORD *)(a1 + 24);
    if ( v74 )
    {
      v75 = v74 - 1;
      v76 = *(_QWORD *)(a1 + 8);
      v73 = 0;
      v18 = v95;
      v77 = 1;
      v78 = v75 & v92;
      v66 = *(_DWORD *)(a1 + 16) + 1;
      result = v76 + 16LL * (v75 & v92);
      v79 = *(_QWORD *)result;
      if ( a2 == *(_QWORD *)result )
        goto LABEL_84;
      while ( v79 != -8 )
      {
        if ( v79 == -16 && !v73 )
          v73 = result;
        v78 = v75 & (v77 + v78);
        result = v76 + 16LL * v78;
        v79 = *(_QWORD *)result;
        if ( a2 == *(_QWORD *)result )
          goto LABEL_84;
        ++v77;
      }
      goto LABEL_92;
    }
    goto LABEL_146;
  }
LABEL_84:
  *(_DWORD *)(a1 + 16) = v66;
  if ( *(_QWORD *)result != -8 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)result = a2;
  *(_DWORD *)(result + 8) = 0;
LABEL_12:
  *(_DWORD *)(result + 8) = v23;
  v29 = *(_QWORD *)(v18 + 8);
  if ( v29 )
  {
    while ( 1 )
    {
      result = (__int64)sub_1648700(v29);
      if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
        break;
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        goto LABEL_15;
    }
LABEL_42:
    v39 = *(_QWORD *)(result + 40);
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      v37 = *(__int64 **)(a2 - 8);
    else
      v37 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    result = sub_1F60340(v39, *v37);
    if ( result )
    {
      v38 = sub_157ED20(result);
      result = sub_1F63C50(a1, v38, v23);
    }
    while ( 1 )
    {
      v29 = *(_QWORD *)(v29 + 8);
      if ( !v29 )
        break;
      result = (__int64)sub_1648700(v29);
      if ( (unsigned __int8)(*(_BYTE *)(result + 16) - 25) <= 9u )
        goto LABEL_42;
    }
  }
LABEL_15:
  v30 = *(_QWORD *)(v15 + 8);
  if ( v30 )
  {
    while ( 1 )
    {
      v31 = sub_1648700(v30);
      result = *((unsigned __int8 *)v31 + 16);
      if ( (_BYTE)result != 34 )
        goto LABEL_27;
      if ( (*((_BYTE *)v31 + 18) & 1) == 0 )
        break;
      v32 = (*((_BYTE *)v31 + 23) & 0x40) != 0 ? (_QWORD *)*(v31 - 1) : &v31[-3 * (*((_DWORD *)v31 + 5) & 0xFFFFFFF)];
      result = v32[3];
      if ( !result )
        break;
      if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        v33 = (*(_BYTE *)(a2 + 23) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
        v34 = *(_QWORD *)(v33 + 24);
        if ( result == v34 )
        {
          if ( v34 )
            break;
        }
      }
LABEL_35:
      v30 = *(_QWORD *)(v30 + 8);
      if ( !v30 )
        return result;
    }
    sub_1F63C50(a1, v31, a3);
    result = *((unsigned __int8 *)v31 + 16);
LABEL_27:
    if ( (_BYTE)result == 73 )
    {
      result = sub_1F5FF70((__int64)v31);
      if ( !result
        || (*(_BYTE *)(a2 + 18) & 1) != 0
        && ((*(_BYTE *)(a2 + 23) & 0x40) == 0
          ? (v35 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))
          : (v35 = *(_QWORD *)(a2 - 8)),
            (v36 = *(_QWORD *)(v35 + 24), result == v36) && v36) )
      {
        result = sub_1F63C50(a1, v31, a3);
      }
    }
    goto LABEL_35;
  }
  return result;
}

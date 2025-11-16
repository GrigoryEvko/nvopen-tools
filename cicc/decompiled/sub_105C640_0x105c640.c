// Function: sub_105C640
// Address: 0x105c640
//
__int64 __fastcall sub_105C640(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // rbx
  __int64 v9; // r12
  _QWORD *v10; // rax
  _QWORD *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  unsigned int v20; // r14d
  __int64 v21; // rax
  __int64 v22; // r12
  _QWORD *v23; // rdi
  _QWORD *v24; // rsi
  bool v25; // al
  _QWORD *v26; // rax
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // r15
  __int64 *v32; // rbx
  __int64 *v33; // r14
  __int64 v34; // r12
  int v35; // edx
  _QWORD *v36; // rdi
  bool v37; // al
  _QWORD *v38; // rax
  _QWORD *v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 *v42; // r10
  int v43; // eax
  __int64 *v44; // rsi
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r10
  int v49; // eax
  unsigned int v50; // edx
  __int64 v51; // rdi
  _QWORD *v52; // rax
  int v53; // esi
  int v54; // r14d
  __int64 v55; // r9
  __int64 v56; // rdi
  __int64 v57; // r8
  unsigned int v58; // edx
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 v61; // rax
  int v62; // esi
  __int64 v63; // r10
  int v64; // edx
  int v65; // r11d
  int v66; // r11d
  unsigned int v67; // ecx
  int v68; // edi
  __int64 v69; // rsi
  int v70; // r10d
  int v71; // r10d
  int v72; // esi
  unsigned int v73; // r11d
  __int64 v74; // rcx
  __int64 v75; // rdi
  __int64 v76; // [rsp+0h] [rbp-B0h]
  __int64 v77; // [rsp+8h] [rbp-A8h]
  __int64 v78; // [rsp+8h] [rbp-A8h]
  char v81; // [rsp+20h] [rbp-90h]
  __int64 v82; // [rsp+30h] [rbp-80h]
  char v83; // [rsp+30h] [rbp-80h]
  unsigned int v84; // [rsp+30h] [rbp-80h]
  int v85; // [rsp+38h] [rbp-78h]
  __int64 *v86; // [rsp+38h] [rbp-78h]
  int v87; // [rsp+38h] [rbp-78h]
  unsigned int v88; // [rsp+38h] [rbp-78h]
  __int64 v89; // [rsp+48h] [rbp-68h] BYREF
  __int64 *v90; // [rsp+50h] [rbp-60h] BYREF
  __int64 v91; // [rsp+58h] [rbp-58h]
  _BYTE v92[80]; // [rsp+60h] [rbp-50h] BYREF

  result = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)result )
    return result;
  v7 = a2;
  do
  {
    v9 = *(_QWORD *)(*(_QWORD *)v7 + 8 * result - 8);
    if ( *(_BYTE *)(a5 + 28) )
    {
      v10 = *(_QWORD **)(a5 + 8);
      v11 = &v10[*(unsigned int *)(a5 + 20)];
      if ( v10 != v11 )
      {
        while ( v9 != *v10 )
        {
          if ( v11 == ++v10 )
            goto LABEL_12;
        }
LABEL_8:
        result = (unsigned int)(*(_DWORD *)(v7 + 8) - 1);
        *(_DWORD *)(v7 + 8) = result;
        continue;
      }
    }
    else if ( sub_C8CA60(a5, *(_QWORD *)(*(_QWORD *)v7 + 8 * result - 8)) )
    {
      goto LABEL_8;
    }
LABEL_12:
    v12 = (__int64 *)sub_E387E0(a3, v9);
    v17 = v12;
    if ( v12 != (__int64 *)a4 && (!a4 || (unsigned __int8)sub_E38870((__int64 *)a4, v12)) )
    {
      do
      {
        v27 = (__int64)v17;
        v17 = (__int64 *)*v17;
      }
      while ( (__int64 *)a4 != v17 );
      v28 = (__int64)&v90;
      v76 = v27;
      v90 = (__int64 *)v92;
      v91 = 0x300000000LL;
      sub_E388C0(v27, (__int64)&v90, v13, 0x300000000LL, v15, v16);
      v30 = (__int64)v90;
      v83 = 0;
      v86 = &v90[(unsigned int)v91];
      if ( v86 == v90 )
        goto LABEL_74;
      v78 = a4;
      v31 = v7;
      v32 = v17;
      v33 = v90;
      while ( 1 )
      {
        while ( 1 )
        {
          v34 = *v33;
          if ( !v32 )
            break;
          v35 = *((_DWORD *)v32 + 18);
          v89 = *v33;
          if ( v35 )
          {
            v46 = *((unsigned int *)v32 + 20);
            v47 = v32[8];
            v48 = v47 + 8 * v46;
            if ( !(_DWORD)v46 )
              goto LABEL_44;
            v49 = v46 - 1;
            v50 = (v46 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
            v28 = v47 + 8LL * v50;
            v51 = *(_QWORD *)v28;
            if ( v34 != *(_QWORD *)v28 )
            {
              v28 = 1;
              while ( v51 != -4096 )
              {
                v29 = (unsigned int)(v28 + 1);
                v50 = v49 & (v28 + v50);
                v28 = v47 + 8LL * v50;
                v51 = *(_QWORD *)v28;
                if ( v34 == *(_QWORD *)v28 )
                  goto LABEL_63;
                v28 = (unsigned int)v29;
              }
              goto LABEL_44;
            }
LABEL_63:
            v37 = v48 != v28;
          }
          else
          {
            v36 = (_QWORD *)v32[11];
            v28 = (__int64)&v36[*((unsigned int *)v32 + 24)];
            v37 = v28 != (_QWORD)sub_1055FB0(v36, v28, &v89);
          }
          if ( v37 )
            break;
LABEL_44:
          if ( v86 == ++v33 )
            goto LABEL_45;
        }
        if ( *(_BYTE *)(a5 + 28) )
        {
          v38 = *(_QWORD **)(a5 + 8);
          v39 = &v38[*(unsigned int *)(a5 + 20)];
          if ( v38 == v39 )
            goto LABEL_54;
          while ( v34 != *v38 )
          {
            if ( v39 == ++v38 )
              goto LABEL_54;
          }
          goto LABEL_44;
        }
        v28 = v34;
        if ( sub_C8CA60(a5, v34) )
          goto LABEL_44;
LABEL_54:
        v41 = *(unsigned int *)(v31 + 8);
        if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(v31 + 12) )
        {
          v28 = v31 + 16;
          sub_C8D5F0(v31, (const void *)(v31 + 16), v41 + 1, 8u, v29, v30);
          v41 = *(unsigned int *)(v31 + 8);
        }
        v83 = 1;
        ++v33;
        *(_QWORD *)(*(_QWORD *)v31 + 8 * v41) = v34;
        ++*(_DWORD *)(v31 + 8);
        if ( v86 == v33 )
        {
LABEL_45:
          v7 = v31;
          a4 = v78;
          if ( v83 )
          {
LABEL_46:
            if ( v90 != (__int64 *)v92 )
              _libc_free(v90, v28);
LABEL_31:
            result = *(unsigned int *)(v7 + 8);
            goto LABEL_9;
          }
LABEL_74:
          --*(_DWORD *)(v7 + 8);
          v28 = a3;
          sub_105BFB0(a1, a3, v76, a5, v29, v30);
          goto LABEL_46;
        }
      }
    }
    v18 = v9 + 48;
    v19 = *(_QWORD *)(v9 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v19 == v9 + 48 )
      goto LABEL_64;
    if ( !v19 )
      BUG();
    v82 = v19 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
      goto LABEL_64;
    v85 = sub_B46E30(v19 - 24);
    if ( !v85 )
      goto LABEL_64;
    v81 = 0;
    v20 = 0;
    v77 = v9;
    do
    {
      while ( 1 )
      {
        v21 = sub_B46EC0(v82, v20);
        v22 = v21;
        if ( !a4 )
          break;
        v90 = (__int64 *)v21;
        if ( *(_DWORD *)(a4 + 72) )
        {
          v18 = *(unsigned int *)(a4 + 80);
          v14 = *(_QWORD *)(a4 + 64);
          v42 = (__int64 *)(v14 + 8 * v18);
          if ( !(_DWORD)v18 )
            goto LABEL_29;
          v43 = v18 - 1;
          v18 = ((_DWORD)v18 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v44 = (__int64 *)(v14 + 8 * v18);
          v45 = *v44;
          if ( v22 != *v44 )
          {
            v62 = 1;
            while ( v45 != -4096 )
            {
              v15 = (unsigned int)(v62 + 1);
              v18 = v43 & (unsigned int)(v62 + v18);
              v44 = (__int64 *)(v14 + 8LL * (unsigned int)v18);
              v45 = *v44;
              if ( v22 == *v44 )
                goto LABEL_60;
              v62 = v15;
            }
            goto LABEL_29;
          }
LABEL_60:
          v25 = v42 != v44;
        }
        else
        {
          v23 = *(_QWORD **)(a4 + 88);
          v24 = &v23[*(unsigned int *)(a4 + 96)];
          v25 = v24 != sub_1055FB0(v23, (__int64)v24, (__int64 *)&v90);
        }
        if ( v25 )
          break;
LABEL_29:
        if ( v85 == ++v20 )
          goto LABEL_30;
      }
      if ( !*(_BYTE *)(a5 + 28) )
      {
        if ( !sub_C8CA60(a5, v22) )
          goto LABEL_49;
        goto LABEL_29;
      }
      v26 = *(_QWORD **)(a5 + 8);
      v18 = (__int64)&v26[*(unsigned int *)(a5 + 20)];
      if ( v26 != (_QWORD *)v18 )
      {
        while ( v22 != *v26 )
        {
          if ( (_QWORD *)v18 == ++v26 )
            goto LABEL_49;
        }
        goto LABEL_29;
      }
LABEL_49:
      v40 = *(unsigned int *)(v7 + 8);
      v14 = *(unsigned int *)(v7 + 12);
      if ( v40 + 1 > v14 )
      {
        sub_C8D5F0(v7, (const void *)(v7 + 16), v40 + 1, 8u, v15, v16);
        v40 = *(unsigned int *)(v7 + 8);
      }
      v18 = *(_QWORD *)v7;
      v81 = 1;
      ++v20;
      *(_QWORD *)(*(_QWORD *)v7 + 8 * v40) = v22;
      ++*(_DWORD *)(v7 + 8);
    }
    while ( v85 != v20 );
LABEL_30:
    v9 = v77;
    if ( v81 )
      goto LABEL_31;
LABEL_64:
    --*(_DWORD *)(v7 + 8);
    if ( *(_BYTE *)(a5 + 28) )
    {
      v52 = *(_QWORD **)(a5 + 8);
      v14 = *(unsigned int *)(a5 + 20);
      v18 = (__int64)&v52[v14];
      if ( v52 == (_QWORD *)v18 )
      {
LABEL_75:
        if ( (unsigned int)v14 >= *(_DWORD *)(a5 + 16) )
          goto LABEL_76;
        *(_DWORD *)(a5 + 20) = v14 + 1;
        *(_QWORD *)v18 = v9;
        ++*(_QWORD *)a5;
      }
      else
      {
        while ( v9 != *v52 )
        {
          if ( (_QWORD *)v18 == ++v52 )
            goto LABEL_75;
        }
      }
    }
    else
    {
LABEL_76:
      sub_C8CC70(a5, v9, v18, v14, v15, v16);
    }
    v53 = *(_DWORD *)(a1 + 88);
    v54 = *(_DWORD *)(a1 + 8);
    v55 = a1 + 64;
    if ( !v53 )
    {
      ++*(_QWORD *)(a1 + 64);
LABEL_102:
      sub_A4A350(v55, 2 * v53);
      v65 = *(_DWORD *)(a1 + 88);
      if ( v65 )
      {
        v66 = v65 - 1;
        v55 = *(_QWORD *)(a1 + 72);
        v67 = v66 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v64 = *(_DWORD *)(a1 + 80) + 1;
        v59 = v55 + 16LL * v67;
        v57 = *(_QWORD *)v59;
        if ( v9 != *(_QWORD *)v59 )
        {
          v68 = 1;
          v69 = 0;
          while ( v57 != -4096 )
          {
            if ( !v69 && v57 == -8192 )
              v69 = v59;
            v67 = v66 & (v68 + v67);
            v59 = v55 + 16LL * v67;
            v57 = *(_QWORD *)v59;
            if ( v9 == *(_QWORD *)v59 )
              goto LABEL_93;
            ++v68;
          }
          if ( v69 )
            v59 = v69;
        }
        goto LABEL_93;
      }
LABEL_125:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
    v56 = *(_QWORD *)(a1 + 72);
    v57 = (unsigned int)(v53 - 1);
    v58 = v57 & (((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9));
    v59 = v56 + 16LL * v58;
    v60 = *(_QWORD *)v59;
    if ( v9 == *(_QWORD *)v59 )
      goto LABEL_71;
    v87 = 1;
    v63 = 0;
    v84 = *(_DWORD *)(a1 + 88);
    while ( v60 != -4096 )
    {
      if ( v60 == -8192 && !v63 )
        v63 = v59;
      v58 = v57 & (v87 + v58);
      v59 = v56 + 16LL * v58;
      v60 = *(_QWORD *)v59;
      if ( v9 == *(_QWORD *)v59 )
        goto LABEL_71;
      ++v87;
    }
    if ( v63 )
      v59 = v63;
    ++*(_QWORD *)(a1 + 64);
    v64 = *(_DWORD *)(a1 + 80) + 1;
    if ( 4 * v64 >= 3 * v84 )
      goto LABEL_102;
    if ( v84 - *(_DWORD *)(a1 + 84) - v64 <= v84 >> 3 )
    {
      v88 = ((unsigned int)v9 >> 4) ^ ((unsigned int)v9 >> 9);
      sub_A4A350(v55, v84);
      v70 = *(_DWORD *)(a1 + 88);
      if ( v70 )
      {
        v71 = v70 - 1;
        v57 = *(_QWORD *)(a1 + 72);
        v72 = 1;
        v73 = v71 & v88;
        v64 = *(_DWORD *)(a1 + 80) + 1;
        v74 = 0;
        v59 = v57 + 16LL * (v71 & v88);
        v75 = *(_QWORD *)v59;
        if ( v9 != *(_QWORD *)v59 )
        {
          while ( v75 != -4096 )
          {
            if ( !v74 && v75 == -8192 )
              v74 = v59;
            v55 = (unsigned int)(v72 + 1);
            v73 = v71 & (v72 + v73);
            v59 = v57 + 16LL * v73;
            v75 = *(_QWORD *)v59;
            if ( v9 == *(_QWORD *)v59 )
              goto LABEL_93;
            ++v72;
          }
          if ( v74 )
            v59 = v74;
        }
        goto LABEL_93;
      }
      goto LABEL_125;
    }
LABEL_93:
    *(_DWORD *)(a1 + 80) = v64;
    if ( *(_QWORD *)v59 != -4096 )
      --*(_DWORD *)(a1 + 84);
    *(_QWORD *)v59 = v9;
    *(_DWORD *)(v59 + 8) = 0;
LABEL_71:
    *(_DWORD *)(v59 + 8) = v54;
    v61 = *(unsigned int *)(a1 + 8);
    if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v61 + 1, 8u, v57, v55);
      v61 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v61) = v9;
    ++*(_DWORD *)(a1 + 8);
    result = *(unsigned int *)(v7 + 8);
LABEL_9:
    ;
  }
  while ( (_DWORD)result );
  return result;
}

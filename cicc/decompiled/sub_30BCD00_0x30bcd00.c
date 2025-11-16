// Function: sub_30BCD00
// Address: 0x30bcd00
//
void __fastcall sub_30BCD00(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 *v3; // r14
  __int64 v4; // r12
  __int64 *v5; // rbx
  __int64 v6; // r15
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // r11
  unsigned int v10; // edi
  __int64 v11; // rcx
  __int64 *v12; // rdx
  int v13; // r10d
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // r8
  unsigned __int64 v17; // r8
  unsigned int v18; // r10d
  _QWORD *v19; // rax
  __int64 v20; // r9
  unsigned int v21; // esi
  __int64 v22; // r12
  int v23; // ecx
  int v24; // ecx
  __int64 v25; // r8
  unsigned int v26; // edi
  int v27; // eax
  __int64 v28; // rsi
  __int64 *v29; // rax
  __int64 v30; // r11
  __int64 *v31; // r14
  int v32; // esi
  int v33; // esi
  __int64 v34; // r8
  unsigned int v35; // ecx
  int v36; // eax
  _QWORD *v37; // rdx
  __int64 v38; // rdi
  int v39; // eax
  int v40; // ecx
  int v41; // ecx
  __int64 v42; // r8
  __int64 *v43; // r10
  int v44; // r13d
  unsigned int v45; // esi
  __int64 v46; // rdi
  __int64 *v47; // rbx
  int v48; // eax
  int v49; // ecx
  int v50; // ecx
  __int64 v51; // rdi
  _QWORD *v52; // r9
  unsigned int v53; // r13d
  int v54; // r10d
  __int64 v55; // rsi
  int v56; // ebx
  _QWORD *v57; // r10
  int v58; // r13d
  __int64 v59; // rt0
  __int64 *v60; // [rsp+10h] [rbp-70h]
  __int64 *v61; // [rsp+10h] [rbp-70h]
  int v62; // [rsp+10h] [rbp-70h]
  __int64 v64; // [rsp+28h] [rbp-58h]
  unsigned int v66; // [rsp+38h] [rbp-48h]
  __int64 v67; // [rsp+38h] [rbp-48h]
  __int64 v68; // [rsp+38h] [rbp-48h]
  __int64 *v69; // [rsp+40h] [rbp-40h]
  __int64 v70[7]; // [rsp+48h] [rbp-38h] BYREF

  v70[0] = a3;
  if ( a1 == a2 || a2 == a1 + 1 )
    return;
  v3 = a1 + 1;
  do
  {
    while ( sub_30BBEA0(v70, *v3, *a1) )
    {
      v4 = *v3;
      v5 = v3 + 1;
      if ( a1 != v3 )
        memmove(a1 + 1, a1, (char *)v3 - (char *)a1);
      ++v3;
      *a1 = v4;
      if ( a2 == v5 )
        return;
    }
    v6 = v70[0];
    v7 = v3;
    v8 = *v3;
    v9 = v7;
    v66 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
    v64 = v70[0] + 96;
    while ( 1 )
    {
      v21 = *(_DWORD *)(v6 + 120);
      v22 = *(v7 - 1);
      v69 = v7;
      if ( v21 )
      {
        v10 = v21 - 1;
        v11 = *(_QWORD *)(v6 + 104);
        v12 = 0;
        v13 = 1;
        v14 = (v21 - 1) & v66;
        v15 = (__int64 *)(v11 + 16LL * v14);
        v16 = *v15;
        if ( v8 == *v15 )
        {
LABEL_10:
          v17 = v15[1];
          goto LABEL_11;
        }
        while ( v16 != -4096 )
        {
          if ( !v12 && v16 == -8192 )
            v12 = v15;
          v14 = v10 & (v13 + v14);
          v15 = (__int64 *)(v11 + 16LL * v14);
          v16 = *v15;
          if ( v8 == *v15 )
            goto LABEL_10;
          ++v13;
        }
        if ( !v12 )
          v12 = v15;
        v39 = *(_DWORD *)(v6 + 112);
        ++*(_QWORD *)(v6 + 96);
        v27 = v39 + 1;
        if ( 4 * v27 < 3 * v21 )
        {
          if ( v21 - *(_DWORD *)(v6 + 116) - v27 > v21 >> 3 )
            goto LABEL_18;
          v61 = v9;
          sub_30BBCC0(v64, v21);
          v40 = *(_DWORD *)(v6 + 120);
          if ( !v40 )
            goto LABEL_91;
          v41 = v40 - 1;
          v42 = *(_QWORD *)(v6 + 104);
          v43 = 0;
          v9 = v61;
          v44 = 1;
          v45 = v41 & v66;
          v27 = *(_DWORD *)(v6 + 112) + 1;
          v12 = (__int64 *)(v42 + 16LL * (v41 & v66));
          v46 = *v12;
          if ( v8 == *v12 )
            goto LABEL_18;
          while ( v46 != -4096 )
          {
            if ( v46 == -8192 && !v43 )
              v43 = v12;
            v45 = v41 & (v44 + v45);
            v12 = (__int64 *)(v42 + 16LL * v45);
            v46 = *v12;
            if ( v8 == *v12 )
              goto LABEL_18;
            ++v44;
          }
          goto LABEL_42;
        }
      }
      else
      {
        ++*(_QWORD *)(v6 + 96);
      }
      v60 = v9;
      sub_30BBCC0(v64, 2 * v21);
      v23 = *(_DWORD *)(v6 + 120);
      if ( !v23 )
        goto LABEL_91;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(v6 + 104);
      v9 = v60;
      v26 = v24 & v66;
      v27 = *(_DWORD *)(v6 + 112) + 1;
      v12 = (__int64 *)(v25 + 16LL * (v24 & v66));
      v28 = *v12;
      if ( v8 == *v12 )
        goto LABEL_18;
      v58 = 1;
      v43 = 0;
      while ( v28 != -4096 )
      {
        if ( !v43 && v28 == -8192 )
          v43 = v12;
        v26 = v24 & (v58 + v26);
        v12 = (__int64 *)(v25 + 16LL * v26);
        v28 = *v12;
        if ( v8 == *v12 )
          goto LABEL_18;
        ++v58;
      }
LABEL_42:
      if ( v43 )
        v12 = v43;
LABEL_18:
      *(_DWORD *)(v6 + 112) = v27;
      if ( *v12 != -4096 )
        --*(_DWORD *)(v6 + 116);
      *v12 = v8;
      v12[1] = 0;
      v21 = *(_DWORD *)(v6 + 120);
      if ( !v21 )
      {
        ++*(_QWORD *)(v6 + 96);
        v29 = v9;
        v30 = v8;
        v31 = v29;
        goto LABEL_22;
      }
      v11 = *(_QWORD *)(v6 + 104);
      v10 = v21 - 1;
      v17 = 0;
LABEL_11:
      v18 = v10 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v19 = (_QWORD *)(v11 + 16LL * v18);
      v20 = *v19;
      if ( v22 != *v19 )
        break;
LABEL_12:
      --v7;
      if ( v17 >= v19[1] )
      {
        v59 = v8;
        v31 = v9;
        v30 = v59;
        goto LABEL_27;
      }
      v7[1] = *v7;
    }
    v62 = 1;
    v37 = 0;
    while ( v20 != -4096 )
    {
      if ( !v37 && v20 == -8192 )
        v37 = v19;
      v18 = v10 & (v62 + v18);
      v19 = (_QWORD *)(v11 + 16LL * v18);
      v20 = *v19;
      if ( v22 == *v19 )
        goto LABEL_12;
      ++v62;
    }
    v47 = v9;
    v30 = v8;
    if ( !v37 )
      v37 = v19;
    v48 = *(_DWORD *)(v6 + 112);
    ++*(_QWORD *)(v6 + 96);
    v31 = v47;
    v36 = v48 + 1;
    if ( 4 * v36 >= 3 * v21 )
    {
LABEL_22:
      v67 = v30;
      sub_30BBCC0(v64, 2 * v21);
      v32 = *(_DWORD *)(v6 + 120);
      if ( !v32 )
        goto LABEL_91;
      v33 = v32 - 1;
      v34 = *(_QWORD *)(v6 + 104);
      v30 = v67;
      v35 = v33 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v36 = *(_DWORD *)(v6 + 112) + 1;
      v37 = (_QWORD *)(v34 + 16LL * v35);
      v38 = *v37;
      if ( v22 != *v37 )
      {
        v56 = 1;
        v57 = 0;
        while ( v38 != -4096 )
        {
          if ( !v57 && v38 == -8192 )
            v57 = v37;
          v35 = v33 & (v56 + v35);
          v37 = (_QWORD *)(v34 + 16LL * v35);
          v38 = *v37;
          if ( v22 == *v37 )
            goto LABEL_24;
          ++v56;
        }
        if ( v57 )
          v37 = v57;
      }
    }
    else if ( v21 - (v36 + *(_DWORD *)(v6 + 116)) <= v21 >> 3 )
    {
      v68 = v30;
      sub_30BBCC0(v64, v21);
      v49 = *(_DWORD *)(v6 + 120);
      if ( v49 )
      {
        v50 = v49 - 1;
        v51 = *(_QWORD *)(v6 + 104);
        v52 = 0;
        v53 = v50 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v30 = v68;
        v54 = 1;
        v36 = *(_DWORD *)(v6 + 112) + 1;
        v37 = (_QWORD *)(v51 + 16LL * v53);
        v55 = *v37;
        if ( v22 != *v37 )
        {
          while ( v55 != -4096 )
          {
            if ( !v52 && v55 == -8192 )
              v52 = v37;
            v53 = v50 & (v54 + v53);
            v37 = (_QWORD *)(v51 + 16LL * v53);
            v55 = *v37;
            if ( v22 == *v37 )
              goto LABEL_24;
            ++v54;
          }
          if ( v52 )
            v37 = v52;
        }
        goto LABEL_24;
      }
LABEL_91:
      ++*(_DWORD *)(v6 + 112);
      BUG();
    }
LABEL_24:
    *(_DWORD *)(v6 + 112) = v36;
    if ( *v37 != -4096 )
      --*(_DWORD *)(v6 + 116);
    *v37 = v22;
    v37[1] = 0;
LABEL_27:
    v3 = v31 + 1;
    *v69 = v30;
  }
  while ( a2 != v3 );
}

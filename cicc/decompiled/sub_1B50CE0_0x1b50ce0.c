// Function: sub_1B50CE0
// Address: 0x1b50ce0
//
bool __fastcall sub_1B50CE0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rdx
  unsigned int v3; // ebx
  __int64 v4; // r15
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // r9
  __int64 v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // r8
  __int64 *v11; // r15
  unsigned int v12; // r12d
  __int64 *v13; // r13
  _QWORD *v14; // r15
  _QWORD *v15; // rax
  __int64 v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  _QWORD *v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned int v22; // eax
  _QWORD *v24; // rdx
  int v25; // r11d
  _QWORD *v26; // rax
  int v27; // edi
  int v28; // edx
  int v29; // r11d
  int v30; // r11d
  __int64 v31; // r9
  unsigned int v32; // ecx
  __int64 v33; // r8
  int v34; // edi
  _QWORD *v35; // rsi
  int v36; // r11d
  int v37; // r11d
  __int64 v38; // r9
  unsigned int v39; // ecx
  int v40; // edi
  __int64 v41; // r8
  __int64 *v42; // [rsp+0h] [rbp-50h]
  __int64 *v43; // [rsp+8h] [rbp-48h]
  __int64 *v44; // [rsp+10h] [rbp-40h]
  unsigned int v45; // [rsp+10h] [rbp-40h]

  v2 = *(__int64 **)(a2 + 16);
  v42 = &v2[*(unsigned int *)(a2 + 24)];
  if ( v42 == v2 )
    return 1;
  v43 = *(__int64 **)(a2 + 16);
  v3 = 0;
  do
  {
    while ( 1 )
    {
      v4 = *v43;
      v5 = *a1;
      v6 = *(_DWORD *)(*a1 + 24);
      if ( !v6 )
      {
        ++*(_QWORD *)v5;
        goto LABEL_41;
      }
      v7 = *(_QWORD *)(v5 + 8);
      LODWORD(v8) = (v6 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v9 = (__int64 *)(v7 + 56LL * (unsigned int)v8);
      v10 = *v9;
      if ( v4 == *v9 )
        break;
      v25 = 1;
      v26 = 0;
      while ( v10 != -8 )
      {
        if ( !v26 && v10 == -16 )
          v26 = v9;
        v8 = (v6 - 1) & ((_DWORD)v8 + v25);
        v9 = (__int64 *)(v7 + 56 * v8);
        v10 = *v9;
        if ( v4 == *v9 )
          goto LABEL_5;
        ++v25;
      }
      v27 = *(_DWORD *)(v5 + 16);
      if ( !v26 )
        v26 = v9;
      ++*(_QWORD *)v5;
      v28 = v27 + 1;
      if ( 4 * (v27 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v5 + 20) - v28 <= v6 >> 3 )
        {
          v45 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
          sub_1B509C0(v5, v6);
          v36 = *(_DWORD *)(v5 + 24);
          if ( !v36 )
          {
LABEL_68:
            ++*(_DWORD *)(v5 + 16);
            BUG();
          }
          v37 = v36 - 1;
          v38 = *(_QWORD *)(v5 + 8);
          v35 = 0;
          v39 = v37 & v45;
          v28 = *(_DWORD *)(v5 + 16) + 1;
          v40 = 1;
          v26 = (_QWORD *)(v38 + 56LL * (v37 & v45));
          v41 = *v26;
          if ( v4 != *v26 )
          {
            while ( v41 != -8 )
            {
              if ( v41 == -16 && !v35 )
                v35 = v26;
              v39 = v37 & (v39 + v40);
              v26 = (_QWORD *)(v38 + 56LL * v39);
              v41 = *v26;
              if ( v4 == *v26 )
                goto LABEL_36;
              ++v40;
            }
LABEL_45:
            if ( v35 )
              v26 = v35;
            goto LABEL_36;
          }
        }
        goto LABEL_36;
      }
LABEL_41:
      sub_1B509C0(v5, 2 * v6);
      v29 = *(_DWORD *)(v5 + 24);
      if ( !v29 )
        goto LABEL_68;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v5 + 8);
      v32 = v30 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v28 = *(_DWORD *)(v5 + 16) + 1;
      v26 = (_QWORD *)(v31 + 56LL * v32);
      v33 = *v26;
      if ( v4 != *v26 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -8 )
        {
          if ( !v35 && v33 == -16 )
            v35 = v26;
          v32 = v30 & (v32 + v34);
          v26 = (_QWORD *)(v31 + 56LL * v32);
          v33 = *v26;
          if ( v4 == *v26 )
            goto LABEL_36;
          ++v34;
        }
        goto LABEL_45;
      }
LABEL_36:
      *(_DWORD *)(v5 + 16) = v28;
      if ( *v26 != -8 )
        --*(_DWORD *)(v5 + 20);
      ++v43;
      *v26 = v4;
      v26[1] = v26 + 3;
      v26[2] = 0x400000000LL;
      if ( v42 == v43 )
        goto LABEL_27;
    }
LABEL_5:
    v11 = (__int64 *)v9[1];
    v44 = &v11[*((unsigned int *)v9 + 4)];
    if ( v11 == v44 )
      goto LABEL_26;
    v12 = v3;
    v13 = (__int64 *)v9[1];
    do
    {
      v17 = *v13;
      v18 = a1[1];
      v19 = *(_QWORD **)(v18 + 16);
      v15 = *(_QWORD **)(v18 + 8);
      if ( v19 == v15 )
      {
        v14 = &v15[*(unsigned int *)(v18 + 28)];
        if ( v15 == v14 )
        {
          v24 = *(_QWORD **)(v18 + 8);
        }
        else
        {
          do
          {
            if ( v17 == *v15 )
              break;
            ++v15;
          }
          while ( v14 != v15 );
          v24 = v14;
        }
      }
      else
      {
        v14 = &v19[*(unsigned int *)(v18 + 24)];
        v15 = sub_16CC9F0(a1[1], *v13);
        if ( v17 == *v15 )
        {
          v20 = *(_QWORD *)(v18 + 16);
          if ( v20 == *(_QWORD *)(v18 + 8) )
            v21 = *(unsigned int *)(v18 + 28);
          else
            v21 = *(unsigned int *)(v18 + 24);
          v24 = (_QWORD *)(v20 + 8 * v21);
        }
        else
        {
          v16 = *(_QWORD *)(v18 + 16);
          if ( v16 != *(_QWORD *)(v18 + 8) )
          {
            v15 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(v18 + 24));
            goto LABEL_10;
          }
          v15 = (_QWORD *)(v16 + 8LL * *(unsigned int *)(v18 + 28));
          v24 = v15;
        }
      }
      while ( v24 != v15 && *v15 >= 0xFFFFFFFFFFFFFFFELL )
        ++v15;
LABEL_10:
      ++v13;
      v12 += v15 == v14;
    }
    while ( v44 != v13 );
    v3 = v12;
LABEL_26:
    ++v43;
  }
  while ( v42 != v43 );
LABEL_27:
  v22 = v3 / *(_DWORD *)(a1[2] + 8);
  if ( v3 % (unsigned __int64)*(unsigned int *)(a1[2] + 8) )
    ++v22;
  return v22 <= 1;
}

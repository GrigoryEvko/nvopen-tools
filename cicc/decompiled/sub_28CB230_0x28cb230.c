// Function: sub_28CB230
// Address: 0x28cb230
//
unsigned int *__fastcall sub_28CB230(__int64 a1)
{
  unsigned int *result; // rax
  __int64 *i; // r12
  _QWORD *v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 v8; // r15
  int v9; // r14d
  __int64 v10; // r13
  int v11; // r14d
  int v12; // edi
  unsigned int j; // edx
  _QWORD *v14; // rax
  unsigned int v15; // edx
  __int64 v16; // rsi
  _BYTE *v17; // rdi
  int v18; // ecx
  __int64 v19; // r8
  int v20; // ecx
  unsigned int v21; // edx
  _QWORD *v22; // rax
  _BYTE *v23; // r9
  __int64 v24; // rax
  char *v25; // rdx
  char v26; // al
  char *v27; // rax
  char v28; // dl
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // r9
  __int64 v32; // rsi
  unsigned int v33; // edx
  unsigned int v34; // r10d
  __int64 v35; // r8
  int v36; // edx
  __int64 v37; // r11
  unsigned int v38; // esi
  __int64 *v39; // rcx
  __int64 v40; // r14
  unsigned int v41; // esi
  __int64 v42; // r8
  unsigned int v43; // ecx
  __int64 v44; // rdi
  int v45; // edx
  unsigned int v46; // r8d
  __int64 *v47; // rcx
  __int64 v48; // r9
  int v49; // eax
  int v50; // ecx
  int v51; // ecx
  int v52; // r10d
  int v53; // r10d
  int v54; // r13d
  __int64 v55; // [rsp+8h] [rbp-48h]
  unsigned int v56; // [rsp+18h] [rbp-38h] BYREF
  unsigned int v57[13]; // [rsp+1Ch] [rbp-34h] BYREF

  result = &v56;
  for ( i = *(__int64 **)a1; *(__int64 **)(a1 + 8) != i; *(_QWORD *)a1 = i )
  {
    v4 = *(_QWORD **)(a1 + 16);
    if ( *(_BYTE *)*v4 == 84 )
    {
      if ( *v4 == *i )
        goto LABEL_35;
      v5 = sub_28C8570(*i);
      if ( v6 == v5 )
        goto LABEL_35;
    }
    v7 = *(_QWORD *)(a1 + 24);
    v8 = i[1];
    v9 = *(_DWORD *)(v7 + 2176);
    v10 = **(_QWORD **)(a1 + 32);
    if ( !v9 )
      goto LABEL_35;
    v11 = v9 - 1;
    v55 = *(_QWORD *)(v7 + 2160);
    v56 = ((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4);
    v57[0] = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
    v12 = 1;
    for ( j = v11 & sub_28052C0(v57, &v56); ; j = v11 & v15 )
    {
      v14 = (_QWORD *)(v55 + 16LL * j);
      if ( v8 == *v14 && v10 == v14[1] )
        break;
      if ( *v14 == -4096 && v14[1] == -4096 )
        goto LABEL_35;
      v15 = v12 + j;
      ++v12;
    }
    v16 = *(_QWORD *)(a1 + 24);
    v17 = (_BYTE *)*i;
    v18 = *(_DWORD *)(v16 + 1456);
    v19 = *(_QWORD *)(v16 + 1440);
    if ( v18 )
    {
      v20 = v18 - 1;
      v21 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v22 = (_QWORD *)(v19 + 16LL * v21);
      v23 = (_BYTE *)*v22;
      if ( v17 == (_BYTE *)*v22 )
      {
LABEL_14:
        v24 = v22[1];
        goto LABEL_15;
      }
      v49 = 1;
      while ( v23 != (_BYTE *)-4096LL )
      {
        v52 = v49 + 1;
        v21 = v20 & (v49 + v21);
        v22 = (_QWORD *)(v19 + 16LL * v21);
        v23 = (_BYTE *)*v22;
        if ( v17 == (_BYTE *)*v22 )
          goto LABEL_14;
        v49 = v52;
      }
    }
    v24 = 0;
LABEL_15:
    if ( *(_QWORD *)(v16 + 1392) != v24 )
    {
      v25 = *(char **)(a1 + 40);
      v26 = *v25;
      if ( *v25 )
        v26 = *v17 <= 0x15u;
      *v25 = v26;
      v27 = *(char **)(a1 + 48);
      v28 = *v27;
      if ( !*v27 )
      {
        v29 = **(_QWORD **)(a1 + 32);
        v28 = 1;
        if ( v8 != v29 )
        {
          v30 = *(_QWORD *)(a1 + 24);
          v31 = *(_QWORD *)(v30 + 8);
          if ( v8 )
          {
            v32 = (unsigned int)(*(_DWORD *)(v8 + 44) + 1);
            v33 = *(_DWORD *)(v8 + 44) + 1;
          }
          else
          {
            v32 = 0;
            v33 = 0;
          }
          v34 = *(_DWORD *)(v31 + 32);
          v35 = 0;
          if ( v33 < v34 )
            v35 = *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8 * v32);
          v36 = *(_DWORD *)(v30 + 1384);
          v37 = *(_QWORD *)(v30 + 1368);
          if ( v36 )
          {
            v38 = (v36 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
            v39 = (__int64 *)(v37 + 16LL * v38);
            v40 = *v39;
            if ( *v39 == v35 )
            {
LABEL_26:
              v41 = *((_DWORD *)v39 + 2);
              goto LABEL_27;
            }
            v51 = 1;
            while ( v40 != -4096 )
            {
              v54 = v51 + 1;
              v38 = (v36 - 1) & (v51 + v38);
              v39 = (__int64 *)(v37 + 16LL * v38);
              v40 = *v39;
              if ( v35 == *v39 )
                goto LABEL_26;
              v51 = v54;
            }
          }
          v41 = 0;
LABEL_27:
          if ( v29 )
          {
            v42 = (unsigned int)(*(_DWORD *)(v29 + 44) + 1);
            v43 = *(_DWORD *)(v29 + 44) + 1;
          }
          else
          {
            v42 = 0;
            v43 = 0;
          }
          v44 = 0;
          if ( v34 > v43 )
            v44 = *(_QWORD *)(*(_QWORD *)(v31 + 24) + 8 * v42);
          if ( v36 )
          {
            v45 = v36 - 1;
            v46 = v45 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
            v47 = (__int64 *)(v37 + 16LL * v46);
            v48 = *v47;
            if ( v44 == *v47 )
            {
LABEL_33:
              v28 = *((_DWORD *)v47 + 2) <= v41;
              goto LABEL_34;
            }
            v50 = 1;
            while ( v48 != -4096 )
            {
              v53 = v50 + 1;
              v46 = v45 & (v50 + v46);
              v47 = (__int64 *)(v37 + 16LL * v46);
              v48 = *v47;
              if ( v44 == *v47 )
                goto LABEL_33;
              v50 = v53;
            }
          }
          v28 = 1;
        }
      }
LABEL_34:
      *v27 = v28;
      result = (unsigned int *)sub_28C86C0(*(_QWORD *)(a1 + 24), *i);
      if ( **(unsigned int ***)(a1 + 16) != result )
        return result;
    }
LABEL_35:
    result = *(unsigned int **)a1;
    i = (__int64 *)(*(_QWORD *)a1 + 16LL);
  }
  return result;
}

// Function: sub_1DE6B80
// Address: 0x1de6b80
//
char __fastcall sub_1DE6B80(__int64 a1, __int64 **a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 v5; // rax
  __int64 *v6; // r11
  __int64 *v7; // r13
  __int64 v11; // rdx
  int v12; // ecx
  int v13; // edi
  __int64 v14; // rsi
  unsigned int v15; // esi
  _QWORD *v16; // r9
  __int64 v17; // r8
  unsigned int v18; // edi
  __int64 v19; // rcx
  __int64 **v20; // rdx
  __int64 v21; // rbx
  __int64 v22; // rbx
  _QWORD *v23; // rdx
  int v24; // eax
  int v25; // eax
  int v26; // esi
  int v27; // esi
  unsigned int v28; // ecx
  __int64 v29; // rdi
  _QWORD *v30; // r15
  int v31; // ecx
  int v32; // ecx
  unsigned int v33; // r15d
  __int64 v34; // rdi
  __int64 v35; // rsi
  __int64 v37; // [rsp+0h] [rbp-50h]
  __int64 v38; // [rsp+0h] [rbp-50h]
  __int64 v39; // [rsp+0h] [rbp-50h]
  __int64 v40; // [rsp+0h] [rbp-50h]
  __int64 *v41; // [rsp+0h] [rbp-50h]
  int v42; // [rsp+8h] [rbp-48h]
  __int64 *v43; // [rsp+8h] [rbp-48h]
  __int64 *v44; // [rsp+8h] [rbp-48h]
  __int64 *v45; // [rsp+8h] [rbp-48h]
  __int64 *v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+10h] [rbp-40h]

  LOBYTE(v5) = a1 + 120;
  v6 = *(__int64 **)(a3 + 96);
  v7 = *(__int64 **)(a3 + 88);
  v47 = a1 + 888;
  if ( v7 != v6 )
  {
    while ( 1 )
    {
      v22 = *v7;
      if ( !a5 )
        goto LABEL_5;
      if ( (*(_BYTE *)(a5 + 8) & 1) != 0 )
        break;
      LODWORD(v5) = *(_DWORD *)(a5 + 24);
      v11 = *(_QWORD *)(a5 + 16);
      v12 = v5 - 1;
      if ( (_DWORD)v5 )
        goto LABEL_4;
LABEL_16:
      if ( v6 == ++v7 )
        return v5;
    }
    v11 = a5 + 16;
    v12 = 15;
LABEL_4:
    v13 = 1;
    LODWORD(v5) = v12 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v14 = *(_QWORD *)(v11 + 8LL * (unsigned int)v5);
    if ( v22 != v14 )
    {
      while ( v14 != -8 )
      {
        LODWORD(v5) = v12 & (v13 + v5);
        v14 = *(_QWORD *)(v11 + 8LL * (unsigned int)v5);
        if ( v22 == v14 )
          goto LABEL_5;
        ++v13;
      }
      goto LABEL_16;
    }
LABEL_5:
    v15 = *(_DWORD *)(a1 + 912);
    if ( v15 )
    {
      LODWORD(v16) = v15 - 1;
      v17 = *(_QWORD *)(a1 + 896);
      v18 = (v15 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v5 = v17 + 16LL * v18;
      v19 = *(_QWORD *)v5;
      if ( v22 == *(_QWORD *)v5 )
      {
        v20 = *(__int64 ***)(v5 + 8);
        LOBYTE(v5) = a2 == v20;
        goto LABEL_8;
      }
      v42 = 1;
      v23 = 0;
      while ( v19 != -8 )
      {
        if ( v19 != -16 || v23 )
          v5 = (unsigned __int64)v23;
        v18 = (unsigned int)v16 & (v42 + v18);
        v41 = (__int64 *)(v17 + 16LL * v18);
        v19 = *v41;
        if ( v22 == *v41 )
        {
          v20 = (__int64 **)v41[1];
          LOBYTE(v5) = a2 == v20;
          goto LABEL_8;
        }
        ++v42;
        v23 = (_QWORD *)v5;
        v5 = v17 + 16LL * v18;
      }
      if ( !v23 )
        v23 = (_QWORD *)v5;
      v24 = *(_DWORD *)(a1 + 904);
      ++*(_QWORD *)(a1 + 888);
      v25 = v24 + 1;
      if ( 4 * v25 < 3 * v15 )
      {
        if ( v15 - *(_DWORD *)(a1 + 908) - v25 <= v15 >> 3 )
        {
          v38 = a4;
          v44 = v6;
          sub_1DE4DF0(v47, v15);
          v31 = *(_DWORD *)(a1 + 912);
          if ( !v31 )
          {
LABEL_68:
            ++*(_DWORD *)(a1 + 904);
            BUG();
          }
          v32 = v31 - 1;
          v16 = 0;
          v6 = v44;
          a4 = v38;
          v33 = v32 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
          v34 = *(_QWORD *)(a1 + 896);
          LODWORD(v17) = 1;
          v25 = *(_DWORD *)(a1 + 904) + 1;
          v23 = (_QWORD *)(v34 + 16LL * v33);
          v35 = *v23;
          if ( v22 != *v23 )
          {
            while ( v35 != -8 )
            {
              if ( v35 == -16 && !v16 )
                v16 = v23;
              v33 = v32 & (v17 + v33);
              v23 = (_QWORD *)(v34 + 16LL * v33);
              v35 = *v23;
              if ( v22 == *v23 )
                goto LABEL_35;
              LODWORD(v17) = v17 + 1;
            }
            if ( v16 )
              v23 = v16;
          }
        }
        goto LABEL_35;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 888);
    }
    v37 = a4;
    v43 = v6;
    sub_1DE4DF0(v47, 2 * v15);
    v26 = *(_DWORD *)(a1 + 912);
    if ( !v26 )
      goto LABEL_68;
    v27 = v26 - 1;
    v17 = *(_QWORD *)(a1 + 896);
    v6 = v43;
    a4 = v37;
    v28 = v27 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
    v25 = *(_DWORD *)(a1 + 904) + 1;
    v23 = (_QWORD *)(v17 + 16LL * v28);
    v29 = *v23;
    if ( v22 != *v23 )
    {
      LODWORD(v16) = 1;
      v30 = 0;
      while ( v29 != -8 )
      {
        if ( !v30 && v29 == -16 )
          v30 = v23;
        v28 = v27 & ((_DWORD)v16 + v28);
        v23 = (_QWORD *)(v17 + 16LL * v28);
        v29 = *v23;
        if ( v22 == *v23 )
          goto LABEL_35;
        LODWORD(v16) = (_DWORD)v16 + 1;
      }
      if ( v30 )
        v23 = v30;
    }
LABEL_35:
    *(_DWORD *)(a1 + 904) = v25;
    if ( *v23 != -8 )
      --*(_DWORD *)(a1 + 908);
    *v23 = v22;
    LOBYTE(v5) = 0;
    v23[1] = 0;
    v20 = 0;
LABEL_8:
    if ( v22 != a4 && !(_BYTE)v5 )
    {
      LODWORD(v5) = *((_DWORD *)v20 + 14);
      if ( (_DWORD)v5 )
      {
        LODWORD(v5) = v5 - 1;
        *((_DWORD *)v20 + 14) = v5;
        if ( !(_DWORD)v5 )
        {
          v21 = **v20;
          if ( *(_BYTE *)(v21 + 180) )
          {
            v5 = *(unsigned int *)(a1 + 384);
            if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 388) )
            {
              v39 = a4;
              v45 = v6;
              sub_16CD150(a1 + 376, (const void *)(a1 + 392), 0, 8, v17, (int)v16);
              a4 = v39;
              v6 = v45;
              v5 = *(unsigned int *)(a1 + 384);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 376) + 8 * v5) = v21;
            ++*(_DWORD *)(a1 + 384);
          }
          else
          {
            v5 = *(unsigned int *)(a1 + 240);
            if ( (unsigned int)v5 >= *(_DWORD *)(a1 + 244) )
            {
              v40 = a4;
              v46 = v6;
              sub_16CD150(a1 + 232, (const void *)(a1 + 248), 0, 8, v17, (int)v16);
              a4 = v40;
              v6 = v46;
              v5 = *(unsigned int *)(a1 + 240);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8 * v5) = v21;
            ++*(_DWORD *)(a1 + 240);
          }
        }
      }
    }
    goto LABEL_16;
  }
  return v5;
}

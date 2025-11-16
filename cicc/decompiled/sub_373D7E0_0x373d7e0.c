// Function: sub_373D7E0
// Address: 0x373d7e0
//
__int64 __fastcall sub_373D7E0(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // rcx
  unsigned int *v7; // r8
  __int64 v8; // r9
  __int64 v9; // r12
  unsigned int v11; // esi
  __int64 *v12; // rdx
  int v13; // r11d
  int v14; // edi
  __int64 *v15; // rax
  __int64 *v16; // rax
  __int64 v17; // r13
  unsigned int v18; // esi
  __int64 v19; // rdi
  int v20; // r11d
  __int64 *v21; // rdx
  unsigned int v22; // ecx
  __int64 *v23; // rax
  __int64 v24; // r9
  __int64 *v25; // rax
  int v26; // eax
  int v27; // ecx
  int v28; // eax
  int v29; // eax
  int v30; // edi
  __int64 v31; // r8
  unsigned int v32; // esi
  __int64 v33; // rax
  int v34; // r10d
  __int64 *v35; // r9
  int v36; // eax
  int v37; // esi
  __int64 v38; // rdi
  __int64 *v39; // r8
  unsigned int v40; // ebx
  int v41; // r9d
  __int64 v42; // rax
  int v43; // eax
  int v44; // esi
  __int64 v45; // rdi
  unsigned int v46; // eax
  int v47; // r11d
  int v48; // eax
  int v49; // esi
  unsigned int v50; // r15d
  __int64 v51; // rdi
  unsigned int *v52; // rax
  int v53; // r10d

  if ( sub_3735ED0(a1, a2) )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  v5 = sub_A777F0(0x30u, (__int64 *)(a1 + 88));
  v9 = v5;
  if ( v5 )
  {
    *(_BYTE *)(v5 + 30) = 0;
    *(_QWORD *)v5 = v5 | 4;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = 0;
    *(_DWORD *)(v5 + 24) = -1;
    *(_WORD *)(v5 + 28) = 11;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
  }
  if ( *(_BYTE *)(a2 + 24) )
  {
    if ( !sub_3734FE0(a1) || (unsigned __int8)sub_321F6A0(*(_QWORD *)(a1 + 208), a1 + 88) )
      v17 = *(_QWORD *)(a1 + 216) + 400LL;
    else
      v17 = a1 + 672;
    v18 = *(_DWORD *)(v17 + 24);
    if ( v18 )
    {
      v19 = *(_QWORD *)(v17 + 8);
      v20 = 1;
      v21 = 0;
      v22 = (v18 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v23 = (__int64 *)(v19 + 16LL * v22);
      v24 = *v23;
      if ( v4 == *v23 )
      {
LABEL_17:
        v25 = v23 + 1;
LABEL_18:
        *v25 = v9;
        return v9;
      }
      while ( v24 != -4096 )
      {
        if ( !v21 && v24 == -8192 )
          v21 = v23;
        v22 = (v18 - 1) & (v20 + v22);
        v23 = (__int64 *)(v19 + 16LL * v22);
        v24 = *v23;
        if ( v4 == *v23 )
          goto LABEL_17;
        ++v20;
      }
      if ( !v21 )
        v21 = v23;
      v26 = *(_DWORD *)(v17 + 16);
      ++*(_QWORD *)v17;
      v27 = v26 + 1;
      if ( 4 * (v26 + 1) < 3 * v18 )
      {
        if ( v18 - *(_DWORD *)(v17 + 20) - v27 > v18 >> 3 )
        {
LABEL_29:
          *(_DWORD *)(v17 + 16) = v27;
          if ( *v21 != -4096 )
            --*(_DWORD *)(v17 + 20);
          *v21 = v4;
          v25 = v21 + 1;
          v21[1] = 0;
          goto LABEL_18;
        }
        sub_373B830(v17, v18);
        v36 = *(_DWORD *)(v17 + 24);
        if ( v36 )
        {
          v37 = v36 - 1;
          v38 = *(_QWORD *)(v17 + 8);
          v39 = 0;
          v40 = (v36 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v41 = 1;
          v27 = *(_DWORD *)(v17 + 16) + 1;
          v21 = (__int64 *)(v38 + 16LL * v40);
          v42 = *v21;
          if ( v4 != *v21 )
          {
            while ( v42 != -4096 )
            {
              if ( v42 == -8192 && !v39 )
                v39 = v21;
              v40 = v37 & (v41 + v40);
              v21 = (__int64 *)(v38 + 16LL * v40);
              v42 = *v21;
              if ( v4 == *v21 )
                goto LABEL_29;
              ++v41;
            }
            if ( v39 )
              v21 = v39;
          }
          goto LABEL_29;
        }
LABEL_99:
        ++*(_DWORD *)(v17 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v17;
    }
    sub_373B830(v17, 2 * v18);
    v29 = *(_DWORD *)(v17 + 24);
    if ( v29 )
    {
      v30 = v29 - 1;
      v31 = *(_QWORD *)(v17 + 8);
      v32 = (v29 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v27 = *(_DWORD *)(v17 + 16) + 1;
      v21 = (__int64 *)(v31 + 16LL * v32);
      v33 = *v21;
      if ( v4 != *v21 )
      {
        v34 = 1;
        v35 = 0;
        while ( v33 != -4096 )
        {
          if ( !v35 && v33 == -8192 )
            v35 = v21;
          v32 = v30 & (v34 + v32);
          v21 = (__int64 *)(v31 + 16LL * v32);
          v33 = *v21;
          if ( v4 == *v21 )
            goto LABEL_29;
          ++v34;
        }
        if ( v35 )
          v21 = v35;
      }
      goto LABEL_29;
    }
    goto LABEL_99;
  }
  if ( !*(_QWORD *)(a2 + 16) )
  {
    v11 = *(_DWORD *)(a1 + 664);
    if ( v11 )
    {
      v8 = v11 - 1;
      v12 = 0;
      v7 = *(unsigned int **)(a1 + 648);
      v13 = 1;
      v14 = v8 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = (__int64 *)&v7[4 * v14];
      v6 = *v15;
      if ( v4 == *v15 )
      {
LABEL_11:
        v16 = v15 + 1;
LABEL_12:
        *v16 = v9;
        goto LABEL_6;
      }
      while ( v6 != -4096 )
      {
        if ( !v12 && v6 == -8192 )
          v12 = v15;
        v14 = v8 & (v13 + v14);
        v15 = (__int64 *)&v7[4 * v14];
        v6 = *v15;
        if ( v4 == *v15 )
          goto LABEL_11;
        ++v13;
      }
      if ( !v12 )
        v12 = v15;
      v28 = *(_DWORD *)(a1 + 656);
      ++*(_QWORD *)(a1 + 640);
      v6 = (unsigned int)(v28 + 1);
      if ( 4 * (int)v6 < 3 * v11 )
      {
        if ( v11 - *(_DWORD *)(a1 + 660) - (unsigned int)v6 > v11 >> 3 )
        {
LABEL_42:
          *(_DWORD *)(a1 + 656) = v6;
          if ( *v12 != -4096 )
            --*(_DWORD *)(a1 + 660);
          *v12 = v4;
          v16 = v12 + 1;
          v12[1] = 0;
          goto LABEL_12;
        }
        sub_373B830(a1 + 640, v11);
        v48 = *(_DWORD *)(a1 + 664);
        if ( v48 )
        {
          v49 = v48 - 1;
          v7 = *(unsigned int **)(a1 + 648);
          v50 = (v48 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
          v6 = (unsigned int)(*(_DWORD *)(a1 + 656) + 1);
          v12 = (__int64 *)&v7[4 * v50];
          v51 = *v12;
          if ( v4 != *v12 )
          {
            v52 = &v7[4 * v50];
            v53 = 1;
            v12 = 0;
            while ( v51 != -4096 )
            {
              if ( v51 == -8192 && !v12 )
                v12 = (__int64 *)v52;
              v8 = (unsigned int)(v53 + 1);
              v50 = v49 & (v53 + v50);
              v52 = &v7[4 * v50];
              v51 = *(_QWORD *)v52;
              if ( v4 == *(_QWORD *)v52 )
              {
                v12 = (__int64 *)&v7[4 * v50];
                goto LABEL_42;
              }
              ++v53;
            }
            if ( !v12 )
              v12 = (__int64 *)v52;
          }
          goto LABEL_42;
        }
LABEL_98:
        ++*(_DWORD *)(a1 + 656);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 640);
    }
    sub_373B830(a1 + 640, 2 * v11);
    v43 = *(_DWORD *)(a1 + 664);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 648);
      v46 = (v43 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v6 = (unsigned int)(*(_DWORD *)(a1 + 656) + 1);
      v12 = (__int64 *)(v45 + 16LL * v46);
      v8 = *v12;
      if ( v4 != *v12 )
      {
        v7 = (unsigned int *)(v45 + 16LL * (v44 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4))));
        v47 = 1;
        v12 = 0;
        while ( v8 != -4096 )
        {
          if ( !v12 && v8 == -8192 )
            v12 = (__int64 *)v7;
          v46 = v44 & (v47 + v46);
          v7 = (unsigned int *)(v45 + 16LL * v46);
          v8 = *(_QWORD *)v7;
          if ( v4 == *(_QWORD *)v7 )
          {
            v12 = (__int64 *)(v45 + 16LL * v46);
            goto LABEL_42;
          }
          ++v47;
        }
        if ( !v12 )
          v12 = (__int64 *)v7;
      }
      goto LABEL_42;
    }
    goto LABEL_98;
  }
LABEL_6:
  sub_373CB80((__int64 *)a1, v9, a2 + 80, v6, v7, v8);
  return v9;
}

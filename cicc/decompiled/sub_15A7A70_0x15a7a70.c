// Function: sub_15A7A70
// Address: 0x15a7a70
//
__int64 __fastcall sub_15A7A70(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        int a8,
        __int64 a9,
        char a10,
        int a11,
        int a12,
        char a13)
{
  int v16; // r15d
  int v17; // r10d
  __int64 v18; // r13
  __int64 v20; // rax
  unsigned int v21; // esi
  __int64 v22; // r14
  __int64 v23; // rcx
  unsigned int v24; // r15d
  unsigned int v25; // edx
  __int64 *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // rdi
  int v30; // r9d
  __int64 *v31; // r8
  int v32; // eax
  int v33; // edx
  int v34; // eax
  int v35; // ecx
  __int64 v36; // rdi
  unsigned int v37; // eax
  __int64 v38; // rsi
  int v39; // r9d
  __int64 *v40; // r8
  int v41; // eax
  int v42; // eax
  __int64 v43; // rsi
  int v44; // r8d
  unsigned int v45; // r15d
  __int64 *v46; // rdi
  __int64 v47; // rcx
  int v48; // [rsp-18h] [rbp-58h]

  if ( !a3 || *a3 == 16 )
  {
    v16 = 0;
    if ( !a13 )
      goto LABEL_4;
  }
  else
  {
    v16 = (int)a3;
    if ( !a13 )
    {
LABEL_4:
      v17 = 0;
      if ( a5 )
        v17 = sub_161FF10(a1, a4, a5);
      v48 = 0;
      goto LABEL_7;
    }
  }
  v17 = 0;
  if ( a5 )
    v17 = sub_161FF10(a1, a4, a5);
  v48 = 1;
LABEL_7:
  v18 = sub_15C37C0(a1, v16, v17, a7, a8, a9, a6, a11, a12, v48, 1);
  if ( a10 )
  {
    v20 = sub_15AB1E0(a3);
    v21 = *(_DWORD *)(a2 + 24);
    v22 = v20;
    if ( v21 )
    {
      v23 = *(_QWORD *)(a2 + 8);
      v24 = ((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4);
      v25 = (v21 - 1) & v24;
      v26 = (__int64 *)(v23 + 32LL * v25);
      v27 = *v26;
      if ( v22 == *v26 )
      {
LABEL_15:
        LODWORD(v28) = *((_DWORD *)v26 + 4);
        if ( (unsigned int)v28 >= *((_DWORD *)v26 + 5) )
        {
          sub_15A6A10((__int64)(v26 + 1), 0);
          v28 = *((unsigned int *)v26 + 4);
          v29 = (__int64 *)(v26[1] + 8 * v28);
        }
        else
        {
          v29 = (__int64 *)(v26[1] + 8LL * (unsigned int)v28);
        }
        if ( !v29 )
        {
LABEL_18:
          *((_DWORD *)v26 + 4) = v28 + 1;
          return v18;
        }
LABEL_28:
        *v29 = v18;
        if ( v18 )
          sub_1623A60(v29, v18, 2);
        LODWORD(v28) = *((_DWORD *)v26 + 4);
        goto LABEL_18;
      }
      v30 = 1;
      v31 = 0;
      while ( v27 != -8 )
      {
        if ( !v31 && v27 == -16 )
          v31 = v26;
        v25 = (v21 - 1) & (v30 + v25);
        v26 = (__int64 *)(v23 + 32LL * v25);
        v27 = *v26;
        if ( v22 == *v26 )
          goto LABEL_15;
        ++v30;
      }
      v32 = *(_DWORD *)(a2 + 16);
      if ( v31 )
        v26 = v31;
      ++*(_QWORD *)a2;
      v33 = v32 + 1;
      if ( 4 * (v32 + 1) < 3 * v21 )
      {
        if ( v21 - *(_DWORD *)(a2 + 20) - v33 > v21 >> 3 )
        {
LABEL_25:
          *(_DWORD *)(a2 + 16) = v33;
          if ( *v26 != -8 )
            --*(_DWORD *)(a2 + 20);
          v29 = v26 + 3;
          *v26 = v22;
          v26[1] = (__int64)(v26 + 3);
          v26[2] = 0x100000000LL;
          goto LABEL_28;
        }
        sub_15A76F0(a2, v21);
        v41 = *(_DWORD *)(a2 + 24);
        if ( v41 )
        {
          v42 = v41 - 1;
          v43 = *(_QWORD *)(a2 + 8);
          v44 = 1;
          v45 = v42 & v24;
          v46 = 0;
          v33 = *(_DWORD *)(a2 + 16) + 1;
          v26 = (__int64 *)(v43 + 32LL * v45);
          v47 = *v26;
          if ( v22 != *v26 )
          {
            while ( v47 != -8 )
            {
              if ( !v46 && v47 == -16 )
                v46 = v26;
              v45 = v42 & (v44 + v45);
              v26 = (__int64 *)(v43 + 32LL * v45);
              v47 = *v26;
              if ( v22 == *v26 )
                goto LABEL_25;
              ++v44;
            }
            if ( v46 )
              v26 = v46;
          }
          goto LABEL_25;
        }
LABEL_61:
        ++*(_DWORD *)(a2 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a2;
    }
    sub_15A76F0(a2, 2 * v21);
    v34 = *(_DWORD *)(a2 + 24);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(a2 + 8);
      v37 = (v34 - 1) & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v33 = *(_DWORD *)(a2 + 16) + 1;
      v26 = (__int64 *)(v36 + 32LL * v37);
      v38 = *v26;
      if ( v22 != *v26 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -8 )
        {
          if ( v38 == -16 && !v40 )
            v40 = v26;
          v37 = v35 & (v39 + v37);
          v26 = (__int64 *)(v36 + 32LL * v37);
          v38 = *v26;
          if ( v22 == *v26 )
            goto LABEL_25;
          ++v39;
        }
        if ( v40 )
          v26 = v40;
      }
      goto LABEL_25;
    }
    goto LABEL_61;
  }
  return v18;
}

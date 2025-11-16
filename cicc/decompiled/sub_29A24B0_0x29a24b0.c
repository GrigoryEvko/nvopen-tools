// Function: sub_29A24B0
// Address: 0x29a24b0
//
__int64 __fastcall sub_29A24B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 *v10; // rax
  __int64 v11; // r14
  unsigned int v12; // esi
  __int64 v13; // r8
  int v14; // r11d
  __int64 *v15; // rdx
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rsi
  unsigned __int64 *v20; // rdx
  int v22; // eax
  int v23; // r9d
  __int64 v24; // r10
  unsigned int v25; // eax
  int v26; // ecx
  __int64 v27; // r8
  int v28; // eax
  int v29; // eax
  int v30; // r8d
  __int64 v31; // rdi
  __int64 *v32; // r9
  __int64 v33; // rbx
  int v34; // eax
  __int64 v35; // rsi
  int v36; // edi
  __int64 *v37; // rsi

  sub_AD0030(a2);
  if ( !*(_BYTE *)(a1 + 28) )
    goto LABEL_13;
  v10 = *(__int64 **)(a1 + 8);
  v7 = *(unsigned int *)(a1 + 20);
  v6 = &v10[v7];
  if ( v10 == v6 )
  {
LABEL_12:
    if ( (unsigned int)v7 >= *(_DWORD *)(a1 + 16) )
    {
LABEL_13:
      sub_C8CC70(a1, a2, (__int64)v6, v7, v8, v9);
      goto LABEL_6;
    }
    *(_DWORD *)(a1 + 20) = v7 + 1;
    *v6 = a2;
    ++*(_QWORD *)a1;
  }
  else
  {
    while ( a2 != *v10 )
    {
      if ( v6 == ++v10 )
        goto LABEL_12;
    }
  }
LABEL_6:
  v11 = *(_QWORD *)(a1 + 448);
  if ( v11 )
  {
    v12 = *(_DWORD *)(v11 + 120);
    if ( v12 )
    {
      v13 = *(_QWORD *)(v11 + 104);
      v14 = 1;
      v15 = 0;
      v16 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v17 = (_QWORD *)(v13 + 16LL * v16);
      v18 = *v17;
      if ( a2 == *v17 )
      {
LABEL_9:
        v19 = v17[1];
        v20 = v17 + 1;
        if ( v19 )
        {
LABEL_10:
          sub_D2F240(**(__int64 ***)(a1 + 456), v19, a3);
          return sub_29A2380(a1, a2);
        }
LABEL_21:
        v19 = sub_D28F90((__int64 *)v11, a2, v20);
        goto LABEL_10;
      }
      while ( v18 != -4096 )
      {
        if ( !v15 && v18 == -8192 )
          v15 = v17;
        v16 = (v12 - 1) & (v14 + v16);
        v17 = (_QWORD *)(v13 + 16LL * v16);
        v18 = *v17;
        if ( a2 == *v17 )
          goto LABEL_9;
        ++v14;
      }
      if ( !v15 )
        v15 = v17;
      v28 = *(_DWORD *)(v11 + 112);
      ++*(_QWORD *)(v11 + 96);
      v26 = v28 + 1;
      if ( 4 * (v28 + 1) < 3 * v12 )
      {
        if ( v12 - *(_DWORD *)(v11 + 116) - v26 > v12 >> 3 )
        {
LABEL_18:
          *(_DWORD *)(v11 + 112) = v26;
          if ( *v15 != -4096 )
            --*(_DWORD *)(v11 + 116);
          *v15 = a2;
          v20 = (unsigned __int64 *)(v15 + 1);
          *v20 = 0;
          goto LABEL_21;
        }
        sub_D25040(v11 + 96, v12);
        v29 = *(_DWORD *)(v11 + 120);
        if ( v29 )
        {
          v30 = v29 - 1;
          v31 = *(_QWORD *)(v11 + 104);
          v32 = 0;
          LODWORD(v33) = (v29 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v26 = *(_DWORD *)(v11 + 112) + 1;
          v34 = 1;
          v15 = (__int64 *)(v31 + 16LL * (unsigned int)v33);
          v35 = *v15;
          if ( a2 != *v15 )
          {
            while ( v35 != -4096 )
            {
              if ( !v32 && v35 == -8192 )
                v32 = v15;
              v33 = v30 & (unsigned int)(v33 + v34);
              v15 = (__int64 *)(v31 + 16 * v33);
              v35 = *v15;
              if ( a2 == *v15 )
                goto LABEL_18;
              ++v34;
            }
            if ( v32 )
              v15 = v32;
          }
          goto LABEL_18;
        }
LABEL_53:
        ++*(_DWORD *)(v11 + 112);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v11 + 96);
    }
    sub_D25040(v11 + 96, 2 * v12);
    v22 = *(_DWORD *)(v11 + 120);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v11 + 104);
      v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(v11 + 112) + 1;
      v15 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v15;
      if ( a2 != *v15 )
      {
        v36 = 1;
        v37 = 0;
        while ( v27 != -4096 )
        {
          if ( !v37 && v27 == -8192 )
            v37 = v15;
          v25 = v23 & (v36 + v25);
          v15 = (__int64 *)(v24 + 16LL * v25);
          v27 = *v15;
          if ( a2 == *v15 )
            goto LABEL_18;
          ++v36;
        }
        if ( v37 )
          v15 = v37;
      }
      goto LABEL_18;
    }
    goto LABEL_53;
  }
  return sub_29A2380(a1, a2);
}

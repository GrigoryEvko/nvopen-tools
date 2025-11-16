// Function: sub_2C47690
// Address: 0x2c47690
//
__int64 __fastcall sub_2C47690(__int64 a1, __int64 a2, __int64 a3)
{
  int v6; // eax
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // r13
  int v14; // eax
  __int16 v15; // ax
  unsigned int v16; // esi
  int v17; // r15d
  __int64 v18; // rdi
  __int64 *v19; // r10
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // rdx
  __int64 *v23; // rax
  _BYTE *v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r14
  _QWORD *v30; // rax
  __int64 v31; // rsi
  int v32; // r8d
  int v33; // eax
  int v34; // edx
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdi
  unsigned int v38; // ecx
  __int64 v39; // rsi
  int v40; // r9d
  __int64 *v41; // r8
  int v42; // eax
  int v43; // ecx
  __int64 v44; // rsi
  int v45; // r8d
  unsigned int v46; // r14d
  __int64 *v47; // rdi
  __int64 v48; // rax

  v6 = *(_DWORD *)(a1 + 584);
  v7 = *(_QWORD *)(a1 + 568);
  if ( !v6 )
    goto LABEL_7;
  v8 = v6 - 1;
  v9 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 != *v10 )
  {
    v14 = 1;
    while ( v11 != -4096 )
    {
      v32 = v14 + 1;
      v9 = v8 & (v14 + v9);
      v10 = (__int64 *)(v7 + 16LL * v9);
      v11 = *v10;
      if ( a2 == *v10 )
        goto LABEL_3;
      v14 = v32;
    }
LABEL_7:
    v15 = *(_WORD *)(a2 + 24);
    if ( v15 )
    {
      if ( v15 == 15 && (v24 = *(_BYTE **)(a2 - 8), *v24 <= 0x1Cu) )
      {
        v12 = sub_2AC42A0(a1, (__int64)v24);
      }
      else
      {
        v25 = (_QWORD *)sub_22077B0(0xA8u);
        v12 = (__int64)v25;
        if ( v25 )
        {
          v12 = (__int64)(v25 + 12);
          sub_2C46B30(v25, a2, a3, v26, v27, v28);
        }
        v29 = *(_QWORD *)a1;
        v30 = (_QWORD *)sub_2BF0490(v12);
        v30[10] = v29;
        v31 = *(_QWORD *)(v29 + 112);
        v30[4] = v29 + 112;
        v31 &= 0xFFFFFFFFFFFFFFF8LL;
        v30[3] = v31 | v30[3] & 7LL;
        *(_QWORD *)(v31 + 8) = v30 + 3;
        *(_QWORD *)(v29 + 112) = *(_QWORD *)(v29 + 112) & 7LL | (unsigned __int64)(v30 + 3);
      }
    }
    else
    {
      v12 = sub_2AC42A0(a1, *(_QWORD *)(a2 + 32));
    }
    v16 = *(_DWORD *)(a1 + 584);
    if ( v16 )
    {
      v17 = 1;
      v18 = *(_QWORD *)(a1 + 568);
      v19 = 0;
      v20 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( a2 == *v21 )
      {
LABEL_11:
        v23 = v21 + 1;
LABEL_12:
        *v23 = v12;
        return v12;
      }
      while ( v22 != -4096 )
      {
        if ( v22 == -8192 && !v19 )
          v19 = v21;
        v20 = (v16 - 1) & (v17 + v20);
        v21 = (__int64 *)(v18 + 16LL * v20);
        v22 = *v21;
        if ( a2 == *v21 )
          goto LABEL_11;
        ++v17;
      }
      if ( !v19 )
        v19 = v21;
      v33 = *(_DWORD *)(a1 + 576);
      ++*(_QWORD *)(a1 + 560);
      v34 = v33 + 1;
      if ( 4 * (v33 + 1) < 3 * v16 )
      {
        if ( v16 - *(_DWORD *)(a1 + 580) - v34 > v16 >> 3 )
        {
LABEL_31:
          *(_DWORD *)(a1 + 576) = v34;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a1 + 580);
          *v19 = a2;
          v23 = v19 + 1;
          v19[1] = 0;
          goto LABEL_12;
        }
        sub_2C2E1E0(a1 + 560, v16);
        v42 = *(_DWORD *)(a1 + 584);
        if ( v42 )
        {
          v43 = v42 - 1;
          v44 = *(_QWORD *)(a1 + 568);
          v45 = 1;
          v46 = (v42 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v34 = *(_DWORD *)(a1 + 576) + 1;
          v47 = 0;
          v19 = (__int64 *)(v44 + 16LL * v46);
          v48 = *v19;
          if ( a2 != *v19 )
          {
            while ( v48 != -4096 )
            {
              if ( !v47 && v48 == -8192 )
                v47 = v19;
              v46 = v43 & (v45 + v46);
              v19 = (__int64 *)(v44 + 16LL * v46);
              v48 = *v19;
              if ( a2 == *v19 )
                goto LABEL_31;
              ++v45;
            }
            if ( v47 )
              v19 = v47;
          }
          goto LABEL_31;
        }
LABEL_58:
        ++*(_DWORD *)(a1 + 576);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 560);
    }
    sub_2C2E1E0(a1 + 560, 2 * v16);
    v35 = *(_DWORD *)(a1 + 584);
    if ( v35 )
    {
      v36 = v35 - 1;
      v37 = *(_QWORD *)(a1 + 568);
      v38 = v36 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v34 = *(_DWORD *)(a1 + 576) + 1;
      v19 = (__int64 *)(v37 + 16LL * v38);
      v39 = *v19;
      if ( a2 != *v19 )
      {
        v40 = 1;
        v41 = 0;
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v41 )
            v41 = v19;
          v38 = v36 & (v40 + v38);
          v19 = (__int64 *)(v37 + 16LL * v38);
          v39 = *v19;
          if ( a2 == *v19 )
            goto LABEL_31;
          ++v40;
        }
        if ( v41 )
          v19 = v41;
      }
      goto LABEL_31;
    }
    goto LABEL_58;
  }
LABEL_3:
  v12 = v10[1];
  if ( !v12 )
    goto LABEL_7;
  return v12;
}

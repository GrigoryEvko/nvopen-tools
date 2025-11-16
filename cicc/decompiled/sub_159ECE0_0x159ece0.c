// Function: sub_159ECE0
// Address: 0x159ece0
//
__int64 __fastcall sub_159ECE0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rdx
  int v15; // r10d
  __int64 *v16; // r11
  int v17; // edi
  int v18; // edx
  unsigned int v20; // r9d
  _QWORD *v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // r9
  int v24; // ecx
  int v25; // ecx
  __int64 v26; // r9
  unsigned int v27; // r8d
  __int64 v28; // rdi
  int v29; // r15d
  __int64 *v30; // r14
  int v31; // ecx
  int v32; // ecx
  __int64 v33; // r9
  unsigned int v34; // r8d
  __int64 v35; // rdi
  int v36; // r14d
  __int64 *v37; // r11
  unsigned int v38; // r10d
  _QWORD *v39; // rsi
  _QWORD *v40; // rdi
  __int64 v41; // r10
  unsigned int v42; // r10d
  _QWORD *v43; // rsi
  _QWORD *v44; // rdi
  __int64 v45; // r10

  v8 = sub_1648A60(24, (unsigned int)a4);
  v9 = v8;
  if ( v8 )
    sub_1594430(v8, a2, a3, a4);
  v10 = *(_DWORD *)(a1 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_28;
  }
  v11 = *(_QWORD *)(a1 + 8);
  v12 = (v10 - 1) & *(_DWORD *)a5;
  v13 = (__int64 *)(v11 + 8LL * v12);
  v14 = *v13;
  if ( *v13 != -8 )
  {
    v15 = 1;
    v16 = 0;
    while ( 1 )
    {
      if ( v14 == -16 )
      {
        if ( !v16 )
          v16 = v13;
      }
      else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)v14 )
      {
        v20 = *(_DWORD *)(v14 + 20) & 0xFFFFFFF;
        if ( *(_QWORD *)(a5 + 24) == v20 )
        {
          if ( !v20 )
            return v9;
          v21 = *(_QWORD **)(a5 + 16);
          v22 = (_QWORD *)(v14 - 24LL * v20);
          v23 = (__int64)&v21[v20];
          while ( *v21 == *v22 )
          {
            ++v21;
            v22 += 3;
            if ( (_QWORD *)v23 == v21 )
              return v9;
          }
        }
      }
      v12 = (v10 - 1) & (v15 + v12);
      v13 = (__int64 *)(v11 + 8LL * v12);
      v14 = *v13;
      if ( *v13 == -8 )
        break;
      ++v15;
    }
    if ( v16 )
      v13 = v16;
  }
  v17 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * (v17 + 1) >= 3 * v10 )
  {
LABEL_28:
    sub_159EB20(a1, 2 * v10);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = v25 & *(_DWORD *)a5;
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v13 = (__int64 *)(v26 + 8LL * v27);
      v28 = *v13;
      if ( *v13 != -8 )
      {
        v29 = 1;
        v30 = 0;
        while ( 1 )
        {
          if ( v28 == -16 )
          {
            if ( !v30 )
              v30 = v13;
          }
          else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)v28 )
          {
            v38 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
            if ( *(_QWORD *)(a5 + 24) == v38 )
            {
              if ( !v38 )
                goto LABEL_14;
              v39 = *(_QWORD **)(a5 + 16);
              v40 = (_QWORD *)(v28 - 24LL * v38);
              v41 = (__int64)&v39[v38];
              while ( *v39 == *v40 )
              {
                ++v39;
                v40 += 3;
                if ( (_QWORD *)v41 == v39 )
                  goto LABEL_14;
              }
            }
          }
          v27 = v25 & (v29 + v27);
          v13 = (__int64 *)(v26 + 8LL * v27);
          v28 = *v13;
          if ( *v13 == -8 )
            break;
          ++v29;
        }
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_14;
    }
    goto LABEL_66;
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v18 <= v10 >> 3 )
  {
    sub_159EB20(a1, v10);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = v32 & *(_DWORD *)a5;
      v18 = *(_DWORD *)(a1 + 16) + 1;
      v13 = (__int64 *)(v33 + 8LL * v34);
      v35 = *v13;
      if ( *v13 != -8 )
      {
        v36 = 1;
        v37 = 0;
        while ( 1 )
        {
          if ( v35 == -16 )
          {
            if ( !v37 )
              v37 = v13;
          }
          else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)v35 )
          {
            v42 = *(_DWORD *)(v35 + 20) & 0xFFFFFFF;
            if ( *(_QWORD *)(a5 + 24) == v42 )
            {
              if ( !v42 )
                goto LABEL_14;
              v43 = *(_QWORD **)(a5 + 16);
              v44 = (_QWORD *)(v35 - 24LL * v42);
              v45 = (__int64)&v43[v42];
              while ( *v43 == *v44 )
              {
                ++v43;
                v44 += 3;
                if ( v43 == (_QWORD *)v45 )
                  goto LABEL_14;
              }
            }
          }
          v34 = v32 & (v36 + v34);
          v13 = (__int64 *)(v33 + 8LL * v34);
          v35 = *v13;
          if ( *v13 == -8 )
            break;
          ++v36;
        }
        if ( v37 )
          v13 = v37;
      }
      goto LABEL_14;
    }
LABEL_66:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v13 != -8 )
    --*(_DWORD *)(a1 + 20);
  *v13 = v9;
  return v9;
}

// Function: sub_AD0DB0
// Address: 0xad0db0
//
__int64 __fastcall sub_AD0DB0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  __int64 *v13; // rdx
  __int64 v14; // rax
  int v15; // r10d
  __int64 *v16; // r11
  int v17; // eax
  int v18; // eax
  unsigned int v20; // r9d
  _QWORD *v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // r9
  int v24; // eax
  int v25; // esi
  __int64 v26; // r9
  unsigned int v27; // r8d
  __int64 v28; // rcx
  int v29; // r13d
  __int64 *v30; // r11
  int v31; // esi
  int v32; // esi
  __int64 v33; // r9
  unsigned int v34; // r8d
  __int64 v35; // rcx
  int v36; // r13d
  unsigned int v37; // r10d
  _QWORD *v38; // rdi
  _QWORD *v39; // rcx
  __int64 v40; // r10
  unsigned int v41; // r10d
  _QWORD *v42; // rdi
  _QWORD *v43; // rcx
  __int64 v44; // r10
  unsigned int v46; // [rsp+Ch] [rbp-34h]

  v46 = a4 & 0x7FFFFFF;
  v8 = sub_BD2C40(24, (unsigned int)a4);
  v9 = v8;
  if ( v8 )
    sub_AC3360(v8, a2, a3, a4, v46);
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
  if ( *v13 != -4096 )
  {
    v15 = 1;
    v16 = 0;
    while ( 1 )
    {
      if ( v14 == -8192 )
      {
        if ( !v16 )
          v16 = v13;
      }
      else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)(v14 + 8) )
      {
        v20 = *(_DWORD *)(v14 + 4) & 0x7FFFFFF;
        if ( *(_QWORD *)(a5 + 24) == v20 )
        {
          if ( !v20 )
            return v9;
          v21 = *(_QWORD **)(a5 + 16);
          v22 = (_QWORD *)(v14 - 32LL * v20);
          v23 = (__int64)&v21[v20];
          while ( *v21 == *v22 )
          {
            ++v21;
            v22 += 4;
            if ( (_QWORD *)v23 == v21 )
              return v9;
          }
        }
      }
      v12 = (v10 - 1) & (v15 + v12);
      v13 = (__int64 *)(v11 + 8LL * v12);
      v14 = *v13;
      if ( *v13 == -4096 )
        break;
      ++v15;
    }
    if ( v16 )
      v13 = v16;
  }
  v17 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v18 = v17 + 1;
  if ( 4 * v18 >= 3 * v10 )
  {
LABEL_28:
    sub_AD0BD0(a1, 2 * v10);
    v24 = *(_DWORD *)(a1 + 24);
    if ( v24 )
    {
      v25 = v24 - 1;
      v26 = *(_QWORD *)(a1 + 8);
      v27 = (v24 - 1) & *(_DWORD *)a5;
      v13 = (__int64 *)(v26 + 8LL * v27);
      v28 = *v13;
      v18 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v13 != -4096 )
      {
        v29 = 1;
        v30 = 0;
        while ( 1 )
        {
          if ( v28 == -8192 )
          {
            if ( !v30 )
              v30 = v13;
          }
          else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)(v28 + 8) )
          {
            v37 = *(_DWORD *)(v28 + 4) & 0x7FFFFFF;
            if ( *(_QWORD *)(a5 + 24) == v37 )
            {
              if ( !v37 )
                goto LABEL_14;
              v38 = *(_QWORD **)(a5 + 16);
              v39 = (_QWORD *)(v28 - 32LL * v37);
              v40 = (__int64)&v38[v37];
              while ( *v38 == *v39 )
              {
                ++v38;
                v39 += 4;
                if ( (_QWORD *)v40 == v38 )
                  goto LABEL_14;
              }
            }
          }
          v27 = v25 & (v29 + v27);
          v13 = (__int64 *)(v26 + 8LL * v27);
          v28 = *v13;
          if ( *v13 == -4096 )
            break;
          ++v29;
        }
LABEL_60:
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_14;
    }
    goto LABEL_63;
  }
  if ( v10 - *(_DWORD *)(a1 + 20) - v18 <= v10 >> 3 )
  {
    sub_AD0BD0(a1, v10);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = v32 & *(_DWORD *)a5;
      v13 = (__int64 *)(v33 + 8LL * v34);
      v35 = *v13;
      v18 = *(_DWORD *)(a1 + 16) + 1;
      if ( *v13 != -4096 )
      {
        v36 = 1;
        v30 = 0;
        while ( 1 )
        {
          if ( v35 == -8192 )
          {
            if ( !v30 )
              v30 = v13;
          }
          else if ( *(_QWORD *)(a5 + 8) == *(_QWORD *)(v35 + 8) )
          {
            v41 = *(_DWORD *)(v35 + 4) & 0x7FFFFFF;
            if ( *(_QWORD *)(a5 + 24) == v41 )
            {
              if ( !v41 )
                goto LABEL_14;
              v42 = *(_QWORD **)(a5 + 16);
              v43 = (_QWORD *)(v35 - 32LL * v41);
              v44 = (__int64)&v42[v41];
              while ( *v42 == *v43 )
              {
                ++v42;
                v43 += 4;
                if ( v42 == (_QWORD *)v44 )
                  goto LABEL_14;
              }
            }
          }
          v34 = v32 & (v36 + v34);
          v13 = (__int64 *)(v33 + 8LL * v34);
          v35 = *v13;
          if ( *v13 == -4096 )
            goto LABEL_60;
          ++v36;
        }
      }
      goto LABEL_14;
    }
LABEL_63:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_14:
  *(_DWORD *)(a1 + 16) = v18;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v13 = v9;
  return v9;
}

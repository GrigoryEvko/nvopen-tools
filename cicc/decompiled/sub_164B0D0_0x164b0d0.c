// Function: sub_164B0D0
// Address: 0x164b0d0
//
__int64 __fastcall sub_164B0D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // r13
  unsigned int v6; // esi
  __int64 v7; // rdi
  __int64 v8; // r8
  unsigned int v9; // ecx
  __int64 v10; // rdx
  int v11; // edx
  int v12; // esi
  __int64 v13; // rdi
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // r8
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // edx
  int v21; // ecx
  __int64 v22; // rdi
  int v23; // r11d
  __int64 v24; // r10
  int v25; // ecx
  int v26; // eax
  int v27; // edx
  __int64 v28; // rdi
  __int64 v29; // r8
  unsigned int v30; // r14d
  int v31; // r9d
  __int64 v32; // rsi
  int v33; // edx
  int v34; // r9d
  int v35; // r10d
  __int64 v36; // r9

  result = sub_16498A0(a1);
  if ( a2 )
  {
    *(_BYTE *)(a1 + 23) |= 0x20u;
    v5 = *(_QWORD *)result;
    v6 = *(_DWORD *)(*(_QWORD *)result + 488LL);
    v7 = *(_QWORD *)result + 464LL;
    if ( v6 )
    {
      v8 = *(_QWORD *)(v5 + 472);
      v9 = (v6 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      result = v8 + 16LL * v9;
      v10 = *(_QWORD *)result;
      if ( a1 == *(_QWORD *)result )
      {
LABEL_4:
        *(_QWORD *)(result + 8) = a2;
        return result;
      }
      v23 = 1;
      v24 = 0;
      while ( v10 != -8 )
      {
        if ( v10 == -16 && !v24 )
          v24 = result;
        v9 = (v6 - 1) & (v23 + v9);
        result = v8 + 16LL * v9;
        v10 = *(_QWORD *)result;
        if ( a1 == *(_QWORD *)result )
          goto LABEL_4;
        ++v23;
      }
      v25 = *(_DWORD *)(v5 + 480);
      if ( v24 )
        result = v24;
      ++*(_QWORD *)(v5 + 464);
      v21 = v25 + 1;
      if ( 4 * v21 < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(v5 + 484) - v21 > v6 >> 3 )
        {
LABEL_13:
          *(_DWORD *)(v5 + 480) = v21;
          if ( *(_QWORD *)result != -8 )
            --*(_DWORD *)(v5 + 484);
          *(_QWORD *)result = a1;
          *(_QWORD *)(result + 8) = 0;
          goto LABEL_4;
        }
        sub_164AF10(v7, v6);
        v26 = *(_DWORD *)(v5 + 488);
        if ( v26 )
        {
          v27 = v26 - 1;
          v28 = *(_QWORD *)(v5 + 472);
          v29 = 0;
          v30 = (v26 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v31 = 1;
          v21 = *(_DWORD *)(v5 + 480) + 1;
          result = v28 + 16LL * v30;
          v32 = *(_QWORD *)result;
          if ( a1 != *(_QWORD *)result )
          {
            while ( v32 != -8 )
            {
              if ( !v29 && v32 == -16 )
                v29 = result;
              v30 = v27 & (v31 + v30);
              result = v28 + 16LL * v30;
              v32 = *(_QWORD *)result;
              if ( a1 == *(_QWORD *)result )
                goto LABEL_13;
              ++v31;
            }
            if ( v29 )
              result = v29;
          }
          goto LABEL_13;
        }
LABEL_52:
        ++*(_DWORD *)(v5 + 480);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v5 + 464);
    }
    sub_164AF10(v7, 2 * v6);
    v17 = *(_DWORD *)(v5 + 488);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v5 + 472);
      v20 = (v17 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v21 = *(_DWORD *)(v5 + 480) + 1;
      result = v19 + 16LL * v20;
      v22 = *(_QWORD *)result;
      if ( a1 != *(_QWORD *)result )
      {
        v35 = 1;
        v36 = 0;
        while ( v22 != -8 )
        {
          if ( !v36 && v22 == -16 )
            v36 = result;
          v20 = v18 & (v35 + v20);
          result = v19 + 16LL * v20;
          v22 = *(_QWORD *)result;
          if ( a1 == *(_QWORD *)result )
            goto LABEL_13;
          ++v35;
        }
        if ( v36 )
          result = v36;
      }
      goto LABEL_13;
    }
    goto LABEL_52;
  }
  if ( (*(_BYTE *)(a1 + 23) & 0x20) != 0 )
  {
    result = *(_QWORD *)result;
    v11 = *(_DWORD *)(result + 488);
    if ( v11 )
    {
      v12 = v11 - 1;
      v13 = *(_QWORD *)(result + 472);
      v14 = (v11 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v15 = (__int64 *)(v13 + 16LL * v14);
      v16 = *v15;
      if ( a1 == *v15 )
      {
LABEL_9:
        *v15 = -16;
        --*(_DWORD *)(result + 480);
        ++*(_DWORD *)(result + 484);
      }
      else
      {
        v33 = 1;
        while ( v16 != -8 )
        {
          v34 = v33 + 1;
          v14 = v12 & (v33 + v14);
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( a1 == *v15 )
            goto LABEL_9;
          v33 = v34;
        }
      }
    }
  }
  *(_BYTE *)(a1 + 23) &= ~0x20u;
  return result;
}

// Function: sub_1540000
// Address: 0x1540000
//
__int64 __fastcall sub_1540000(__int64 a1)
{
  __int64 v2; // rcx
  unsigned int v3; // esi
  int v4; // r9d
  int v5; // eax
  __int64 v6; // r8
  __int64 v7; // rdi
  int v8; // ecx
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r10
  __int64 v12; // rdx
  unsigned int v13; // esi
  int v14; // r9d
  int v15; // eax
  __int64 v16; // r8
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // r8
  int v26; // eax
  __int64 v27; // rdi
  int v28; // ecx
  __int64 v29; // r9
  unsigned int v30; // edx
  __int64 *v31; // rax
  __int64 v32; // r10
  __int64 v33; // rdx
  unsigned __int64 v34; // rsi
  unsigned __int64 v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rcx
  unsigned __int64 v38; // rsi
  unsigned __int64 v39; // rax
  __int64 result; // rax
  int v41; // eax
  int v42; // r11d
  int v43; // eax
  int v44; // r11d
  int v45; // eax
  int v46; // r11d
  __int64 v47; // rax

  v2 = *(_QWORD *)(a1 + 112);
  v3 = *(_DWORD *)(a1 + 536);
  if ( v3 != (unsigned int)((*(_QWORD *)(a1 + 120) - v2) >> 4) )
  {
    v4 = (*(_QWORD *)(a1 + 120) - v2) >> 4;
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 104);
      if ( v5 )
      {
        v6 = *(_QWORD *)(a1 + 88);
        v7 = *(_QWORD *)(v2 + 16LL * v3);
        v8 = v5 - 1;
        v9 = (v5 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
        {
LABEL_6:
          *v10 = -16;
          --*(_DWORD *)(a1 + 96);
          ++*(_DWORD *)(a1 + 100);
        }
        else
        {
          v43 = 1;
          while ( v11 != -8 )
          {
            v44 = v43 + 1;
            v9 = v8 & (v43 + v9);
            v10 = (__int64 *)(v6 + 16LL * v9);
            v11 = *v10;
            if ( v7 == *v10 )
              goto LABEL_6;
            v43 = v44;
          }
        }
      }
      if ( v4 == ++v3 )
        break;
      v2 = *(_QWORD *)(a1 + 112);
    }
  }
  v12 = *(_QWORD *)(a1 + 208);
  v13 = *(_DWORD *)(a1 + 540);
  v14 = (*(_QWORD *)(a1 + 216) - v12) >> 3;
  if ( v13 != v14 )
  {
    while ( 1 )
    {
      v15 = *(_DWORD *)(a1 + 280);
      if ( v15 )
      {
        v16 = *(_QWORD *)(a1 + 264);
        v17 = *(_QWORD *)(v12 + 8LL * v13);
        v18 = v15 - 1;
        v19 = (v15 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v20 = (__int64 *)(v16 + 16LL * v19);
        v21 = *v20;
        if ( v17 == *v20 )
        {
LABEL_13:
          *v20 = -8;
          --*(_DWORD *)(a1 + 272);
          ++*(_DWORD *)(a1 + 276);
        }
        else
        {
          v45 = 1;
          while ( v21 != -4 )
          {
            v46 = v45 + 1;
            v19 = v18 & (v45 + v19);
            v20 = (__int64 *)(v16 + 16LL * v19);
            v21 = *v20;
            if ( v17 == *v20 )
              goto LABEL_13;
            v45 = v46;
          }
        }
      }
      if ( v14 == ++v13 )
        break;
      v12 = *(_QWORD *)(a1 + 208);
    }
  }
  v22 = *(_QWORD *)(a1 + 512);
  v23 = (*(_QWORD *)(a1 + 520) - v22) >> 3;
  if ( (_DWORD)v23 )
  {
    v24 = 0;
    v25 = 8LL * (unsigned int)(v23 - 1);
    while ( 1 )
    {
      v26 = *(_DWORD *)(a1 + 104);
      if ( v26 )
      {
        v27 = *(_QWORD *)(v22 + v24);
        v28 = v26 - 1;
        v29 = *(_QWORD *)(a1 + 88);
        v30 = (v26 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
        v31 = (__int64 *)(v29 + 16LL * v30);
        v32 = *v31;
        if ( *v31 == v27 )
        {
LABEL_20:
          *v31 = -16;
          --*(_DWORD *)(a1 + 96);
          ++*(_DWORD *)(a1 + 100);
        }
        else
        {
          v41 = 1;
          while ( v32 != -8 )
          {
            v42 = v41 + 1;
            v30 = v28 & (v41 + v30);
            v31 = (__int64 *)(v29 + 16LL * v30);
            v32 = *v31;
            if ( v27 == *v31 )
              goto LABEL_20;
            v41 = v42;
          }
        }
      }
      if ( v25 == v24 )
        break;
      v22 = *(_QWORD *)(a1 + 512);
      v24 += 8;
    }
  }
  v33 = *(_QWORD *)(a1 + 112);
  v34 = *(unsigned int *)(a1 + 536);
  v35 = (*(_QWORD *)(a1 + 120) - v33) >> 4;
  if ( v34 > v35 )
  {
    sub_153FCA0((const __m128i **)(a1 + 112), v34 - v35);
  }
  else if ( v34 < v35 )
  {
    v36 = v33 + 16 * v34;
    if ( *(_QWORD *)(a1 + 120) != v36 )
      *(_QWORD *)(a1 + 120) = v36;
  }
  v37 = *(_QWORD *)(a1 + 208);
  v38 = *(unsigned int *)(a1 + 540);
  v39 = (*(_QWORD *)(a1 + 216) - v37) >> 3;
  if ( v38 > v39 )
  {
    sub_153FE50(a1 + 208, v38 - v39);
  }
  else if ( v38 < v39 )
  {
    v47 = v37 + 8 * v38;
    if ( *(_QWORD *)(a1 + 216) != v47 )
      *(_QWORD *)(a1 + 216) = v47;
  }
  result = *(_QWORD *)(a1 + 512);
  if ( result != *(_QWORD *)(a1 + 520) )
    *(_QWORD *)(a1 + 520) = result;
  *(_DWORD *)(a1 + 544) = 0;
  return result;
}

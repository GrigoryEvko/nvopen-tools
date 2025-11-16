// Function: sub_1E836C0
// Address: 0x1e836c0
//
__int64 __fastcall sub_1E836C0(unsigned int *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int16 v8; // ax
  int v9; // eax
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // r11d
  __int64 *v13; // rdx
  unsigned int v14; // ecx
  __int64 *v15; // rax
  __int64 v16; // r9
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // rdi
  int v25; // r10d
  __int64 *v26; // r9
  int v27; // eax
  int v28; // eax
  __int64 v29; // rdi
  int v30; // r9d
  unsigned int v31; // r14d
  __int64 *v32; // r8
  __int64 v33; // rsi

  v6 = *(_QWORD *)a1;
  v8 = **(_WORD **)(*(_QWORD *)a1 + 16LL);
  switch ( v8 )
  {
    case 0:
    case 8:
    case 10:
    case 14:
    case 15:
    case 45:
      break;
    default:
      switch ( v8 )
      {
        case 2:
        case 3:
        case 4:
        case 6:
        case 9:
        case 12:
        case 13:
        case 17:
        case 18:
          goto LABEL_4;
        default:
          v9 = sub_1F4BB70(a5, *(_QWORD *)a1, a1[2], a2, a1[3]);
          v6 = *(_QWORD *)a1;
          a3 += v9;
          break;
      }
      break;
  }
LABEL_4:
  v10 = *(_DWORD *)(a4 + 24);
  if ( !v10 )
  {
    ++*(_QWORD *)a4;
    goto LABEL_23;
  }
  v11 = *(_QWORD *)(a4 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v10 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v15 = (__int64 *)(v11 + 16LL * v14);
  v16 = *v15;
  if ( v6 == *v15 )
  {
LABEL_6:
    if ( a3 > *((_DWORD *)v15 + 2) )
      *((_DWORD *)v15 + 2) = a3;
    return 0;
  }
  while ( v16 != -8 )
  {
    if ( !v13 && v16 == -16 )
      v13 = v15;
    v14 = (v10 - 1) & (v12 + v14);
    v15 = (__int64 *)(v11 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == v6 )
      goto LABEL_6;
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v18 = *(_DWORD *)(a4 + 16);
  ++*(_QWORD *)a4;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v10 )
  {
LABEL_23:
    sub_1E83500(a4, 2 * v10);
    v20 = *(_DWORD *)(a4 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a4 + 8);
      v23 = (v20 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v19 = *(_DWORD *)(a4 + 16) + 1;
      v13 = (__int64 *)(v22 + 16LL * v23);
      v24 = *v13;
      if ( *v13 != v6 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v26 )
            v26 = v13;
          v23 = v21 & (v25 + v23);
          v13 = (__int64 *)(v22 + 16LL * v23);
          v24 = *v13;
          if ( *v13 == v6 )
            goto LABEL_19;
          ++v25;
        }
        if ( v26 )
          v13 = v26;
      }
      goto LABEL_19;
    }
    goto LABEL_46;
  }
  if ( v10 - *(_DWORD *)(a4 + 20) - v19 <= v10 >> 3 )
  {
    sub_1E83500(a4, v10);
    v27 = *(_DWORD *)(a4 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a4 + 8);
      v30 = 1;
      v31 = v28 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v32 = 0;
      v19 = *(_DWORD *)(a4 + 16) + 1;
      v13 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v13;
      if ( *v13 != v6 )
      {
        while ( v33 != -8 )
        {
          if ( !v32 && v33 == -16 )
            v32 = v13;
          v31 = v28 & (v30 + v31);
          v13 = (__int64 *)(v29 + 16LL * v31);
          v33 = *v13;
          if ( v6 == *v13 )
            goto LABEL_19;
          ++v30;
        }
        if ( v32 )
          v13 = v32;
      }
      goto LABEL_19;
    }
LABEL_46:
    ++*(_DWORD *)(a4 + 16);
    BUG();
  }
LABEL_19:
  *(_DWORD *)(a4 + 16) = v19;
  if ( *v13 != -8 )
    --*(_DWORD *)(a4 + 20);
  *v13 = v6;
  *((_DWORD *)v13 + 2) = a3;
  return 1;
}

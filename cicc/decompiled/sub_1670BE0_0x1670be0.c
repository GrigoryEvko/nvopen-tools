// Function: sub_1670BE0
// Address: 0x1670be0
//
__int64 __fastcall sub_1670BE0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r12d
  __int64 v7; // rdi
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // r10d
  _QWORD *v16; // r15
  int v17; // eax
  int v18; // edx
  char v19; // al
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // r15
  _QWORD *v24; // rax
  char v25; // dl
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // eax
  int v31; // ecx
  __int64 v32; // rdi
  unsigned int v33; // eax
  __int64 v34; // rsi
  int v35; // r9d
  _QWORD *v36; // r8
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  unsigned int v41; // r12d
  _QWORD *v42; // rdi
  __int64 v43; // rcx
  _QWORD *v44; // rsi
  unsigned int v45; // edi
  _QWORD *v46; // rcx
  int v47; // ecx
  int v48; // esi

  if ( *(_BYTE *)(a2 + 8) != *(_BYTE *)(a3 + 8) )
    return 0;
  v7 = a1 + 8;
  v9 = *(_DWORD *)(a1 + 32);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_45;
  }
  v10 = *(_QWORD *)(a1 + 16);
  v3 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
  v11 = (v9 - 1) & v3;
  v12 = (_QWORD *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( *v12 != a3 )
  {
    v15 = 1;
    v16 = 0;
    while ( v13 != -8 )
    {
      if ( v13 == -16 && !v16 )
        v16 = v12;
      v11 = (v9 - 1) & (v15 + v11);
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == a3 )
        goto LABEL_6;
      ++v15;
    }
    if ( !v16 )
      v16 = v12;
    v17 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v18 = v17 + 1;
    if ( 4 * (v17 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(a1 + 28) - v18 > v9 >> 3 )
      {
LABEL_14:
        *(_DWORD *)(a1 + 24) = v18;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 28);
        *v16 = a3;
        v16[1] = 0;
        goto LABEL_17;
      }
      sub_1670A20(v7, v9);
      v37 = *(_DWORD *)(a1 + 32);
      if ( v37 )
      {
        v38 = v37 - 1;
        v39 = *(_QWORD *)(a1 + 16);
        v40 = 1;
        v41 = v38 & v3;
        v18 = *(_DWORD *)(a1 + 24) + 1;
        v42 = 0;
        v16 = (_QWORD *)(v39 + 16LL * v41);
        v43 = *v16;
        if ( *v16 != a3 )
        {
          while ( v43 != -8 )
          {
            if ( v43 == -16 && !v42 )
              v42 = v16;
            v41 = v38 & (v40 + v41);
            v16 = (_QWORD *)(v39 + 16LL * v41);
            v43 = *v16;
            if ( *v16 == a3 )
              goto LABEL_14;
            ++v40;
          }
          if ( v42 )
            v16 = v42;
        }
        goto LABEL_14;
      }
LABEL_94:
      JUMPOUT(0x41A61A);
    }
LABEL_45:
    sub_1670A20(v7, 2 * v9);
    v30 = *(_DWORD *)(a1 + 32);
    if ( v30 )
    {
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 16);
      v33 = (v30 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = *(_DWORD *)(a1 + 24) + 1;
      v16 = (_QWORD *)(v32 + 16LL * v33);
      v34 = *v16;
      if ( *v16 != a3 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -8 )
        {
          if ( v34 == -16 && !v36 )
            v36 = v16;
          v33 = v31 & (v35 + v33);
          v16 = (_QWORD *)(v32 + 16LL * v33);
          v34 = *v16;
          if ( *v16 == a3 )
            goto LABEL_14;
          ++v35;
        }
        if ( v36 )
          v16 = v36;
      }
      goto LABEL_14;
    }
    goto LABEL_94;
  }
LABEL_6:
  v14 = v12[1];
  if ( v14 )
  {
    LOBYTE(v3) = v14 == a2;
    return v3;
  }
  v16 = v12;
LABEL_17:
  if ( a2 == a3 )
  {
    v16[1] = a2;
    return 1;
  }
  if ( *(_BYTE *)(a3 + 8) != 13 )
  {
LABEL_21:
    if ( *(_DWORD *)(a2 + 12) == *(_DWORD *)(a3 + 12) )
    {
      v19 = *(_BYTE *)(a2 + 8);
      if ( v19 != 11 )
      {
        switch ( v19 )
        {
          case 15:
            if ( *(_DWORD *)(a2 + 8) >> 8 == *(_DWORD *)(a3 + 8) >> 8 )
            {
LABEL_25:
              v16[1] = a2;
              v20 = *(unsigned int *)(a1 + 48);
              if ( (unsigned int)v20 >= *(_DWORD *)(a1 + 52) )
              {
                sub_16CD150(a1 + 40, a1 + 56, 0, 8);
                v20 = *(unsigned int *)(a1 + 48);
              }
              v21 = 0;
              *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v20) = a3;
              ++*(_DWORD *)(a1 + 48);
              v22 = *(unsigned int *)(a3 + 12);
              v23 = 8 * v22;
              if ( !(_DWORD)v22 )
                return 1;
              while ( (unsigned __int8)sub_1670BE0(
                                         a1,
                                         *(_QWORD *)(*(_QWORD *)(a2 + 16) + v21),
                                         *(_QWORD *)(*(_QWORD *)(a3 + 16) + v21)) )
              {
                v21 += 8;
                if ( v23 == v21 )
                  return 1;
              }
            }
            break;
          case 12:
            if ( (*(_DWORD *)(a3 + 8) >> 8 != 0) == (*(_DWORD *)(a2 + 8) >> 8 != 0) )
              goto LABEL_25;
            break;
          case 13:
            v47 = *(_DWORD *)(a2 + 8);
            v48 = *(_DWORD *)(a3 + 8);
            if ( ((v48 & 0x400) != 0) == ((v47 & 0x400) != 0) && ((v48 & 0x200) != 0) == ((v47 & 0x200) != 0) )
              goto LABEL_25;
            break;
          default:
            if ( ((v19 - 14) & 0xFD) != 0 || *(_QWORD *)(a3 + 32) == *(_QWORD *)(a2 + 32) )
              goto LABEL_25;
            break;
        }
      }
    }
    return 0;
  }
  v3 = (*(_DWORD *)(a3 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(a3 + 8) & 0x100) != 0 )
  {
    v3 = (*(_DWORD *)(a2 + 8) & 0x100) == 0;
    if ( (*(_DWORD *)(a2 + 8) & 0x100) != 0 )
      goto LABEL_21;
    v24 = *(_QWORD **)(a1 + 480);
    if ( *(_QWORD **)(a1 + 488) != v24 )
      goto LABEL_33;
    v44 = &v24[*(unsigned int *)(a1 + 500)];
    v45 = *(_DWORD *)(a1 + 500);
    if ( v24 != v44 )
    {
      v46 = 0;
      while ( a2 != *v24 )
      {
        if ( *v24 == -2 )
          v46 = v24;
        if ( v44 == ++v24 )
        {
          if ( !v46 )
            goto LABEL_81;
          *v46 = a2;
          --*(_DWORD *)(a1 + 504);
          ++*(_QWORD *)(a1 + 472);
          goto LABEL_34;
        }
      }
      return 0;
    }
LABEL_81:
    if ( v45 < *(_DWORD *)(a1 + 496) )
    {
      *(_DWORD *)(a1 + 500) = v45 + 1;
      *v44 = a2;
      ++*(_QWORD *)(a1 + 472);
    }
    else
    {
LABEL_33:
      sub_16CCBA0(a1 + 472, a2);
      if ( !v25 )
        return 0;
    }
LABEL_34:
    v26 = *(unsigned int *)(a1 + 336);
    if ( (unsigned int)v26 >= *(_DWORD *)(a1 + 340) )
    {
      sub_16CD150(a1 + 328, a1 + 344, 0, 8);
      v26 = *(unsigned int *)(a1 + 336);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8 * v26) = a3;
    v27 = *(unsigned int *)(a1 + 48);
    ++*(_DWORD *)(a1 + 336);
    if ( (unsigned int)v27 >= *(_DWORD *)(a1 + 52) )
    {
      sub_16CD150(a1 + 40, a1 + 56, 0, 8);
      v27 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v27) = a3;
    v28 = *(unsigned int *)(a1 + 192);
    ++*(_DWORD *)(a1 + 48);
    if ( (unsigned int)v28 >= *(_DWORD *)(a1 + 196) )
    {
      sub_16CD150(a1 + 184, a1 + 200, 0, 8);
      v28 = *(unsigned int *)(a1 + 192);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v28) = a2;
    ++*(_DWORD *)(a1 + 192);
    v16[1] = a2;
  }
  else
  {
    v16[1] = a2;
    v29 = *(unsigned int *)(a1 + 48);
    if ( (unsigned int)v29 >= *(_DWORD *)(a1 + 52) )
    {
      sub_16CD150(a1 + 40, a1 + 56, 0, 8);
      v29 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v29) = a3;
    ++*(_DWORD *)(a1 + 48);
  }
  return v3;
}

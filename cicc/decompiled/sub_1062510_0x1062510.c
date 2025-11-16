// Function: sub_1062510
// Address: 0x1062510
//
__int64 __fastcall sub_1062510(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  unsigned int v6; // esi
  __int64 v8; // rbx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdi
  int v12; // r11d
  __int64 *v13; // r9
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // r10
  __int64 *v17; // r15
  __int64 v18; // rax
  char v19; // al
  unsigned __int8 v20; // dl
  __int64 v21; // rax
  unsigned __int64 v22; // rcx
  __int64 v23; // rbx
  __int64 v24; // rax
  int v25; // edx
  __int64 v26; // rbx
  __int64 v27; // r12
  int v28; // eax
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  __int64 v33; // rbx
  __int64 v34; // rax
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rbx
  char v38; // dl
  int v39; // eax
  int v40; // edx
  __int64 v41; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v42; // [rsp+18h] [rbp-38h] BYREF

  v41 = a3;
  if ( *(_BYTE *)(a2 + 8) != *(_BYTE *)(a3 + 8) )
    goto LABEL_2;
  v6 = *(_DWORD *)(a1 + 32);
  v3 = a1 + 8;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 8);
    v42 = 0;
    goto LABEL_49;
  }
  v8 = a3;
  v9 = a3;
  v10 = v6 - 1;
  v11 = *(_QWORD *)(a1 + 16);
  v12 = 1;
  v13 = 0;
  v14 = (unsigned int)v10 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v11 + 16LL * (unsigned int)v14);
  v16 = *v15;
  if ( v8 != *v15 )
  {
    while ( v16 != -4096 )
    {
      if ( v16 == -8192 && !v13 )
        v13 = v15;
      v14 = (unsigned int)v10 & (v12 + (_DWORD)v14);
      v15 = (__int64 *)(v11 + 16LL * (unsigned int)v14);
      v16 = *v15;
      if ( v8 == *v15 )
        goto LABEL_6;
      ++v12;
    }
    if ( !v13 )
      v13 = v15;
    v28 = *(_DWORD *)(a1 + 24);
    ++*(_QWORD *)(a1 + 8);
    v14 = (unsigned int)(v28 + 1);
    v42 = v13;
    if ( 4 * (int)v14 < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 28) - (unsigned int)v14 > v6 >> 3 )
      {
LABEL_32:
        *(_DWORD *)(a1 + 24) = v14;
        if ( *v13 != -4096 )
          --*(_DWORD *)(a1 + 28);
        *v13 = v9;
        v17 = v13 + 1;
        v8 = v41;
        v13[1] = 0;
        goto LABEL_7;
      }
LABEL_50:
      sub_1062330(v3, v6);
      sub_1061BC0(v3, &v41, &v42);
      v9 = v41;
      v13 = v42;
      v14 = (unsigned int)(*(_DWORD *)(a1 + 24) + 1);
      goto LABEL_32;
    }
LABEL_49:
    v6 *= 2;
    goto LABEL_50;
  }
LABEL_6:
  v17 = v15 + 1;
  v18 = v15[1];
  LOBYTE(v3) = a2 == v18;
  if ( v18 )
    return (unsigned int)v3;
LABEL_7:
  if ( v8 == a2 )
  {
    *v17 = v8;
    LODWORD(v3) = 1;
    return (unsigned int)v3;
  }
  v19 = *(_BYTE *)(v8 + 8);
  if ( v19 != 15 )
  {
LABEL_11:
    if ( *(_DWORD *)(v8 + 12) == *(_DWORD *)(a2 + 12) )
    {
      v20 = *(_BYTE *)(a2 + 8);
      if ( v20 != 12 )
      {
        switch ( v20 )
        {
          case 0xEu:
            if ( *(_DWORD *)(a2 + 8) >> 8 == *(_DWORD *)(v8 + 8) >> 8 )
            {
LABEL_15:
              *v17 = a2;
              v21 = *(unsigned int *)(a1 + 48);
              v22 = *(unsigned int *)(a1 + 52);
              v23 = v41;
              if ( v21 + 1 > v22 )
              {
                sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v21 + 1, 8u, v10, (__int64)v13);
                v21 = *(unsigned int *)(a1 + 48);
              }
              *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v21) = v23;
              v24 = v41;
              ++*(_DWORD *)(a1 + 48);
              v25 = *(_DWORD *)(v24 + 12);
              if ( !v25 )
              {
LABEL_60:
                LODWORD(v3) = 1;
                return (unsigned int)v3;
              }
              v26 = 0;
              v27 = 8LL * (unsigned int)(v25 - 1);
              while ( (unsigned __int8)sub_1062510(
                                         a1,
                                         *(_QWORD *)(*(_QWORD *)(a2 + 16) + v26),
                                         *(_QWORD *)(*(_QWORD *)(v24 + 16) + v26),
                                         v22,
                                         v10,
                                         v13) )
              {
                if ( v26 == v27 )
                  goto LABEL_60;
                v24 = v41;
                v26 += 8;
              }
            }
            break;
          case 0xDu:
            if ( (*(_DWORD *)(a2 + 8) >> 8 != 0) == (*(_DWORD *)(v8 + 8) >> 8 != 0) )
              goto LABEL_15;
            break;
          case 0xFu:
            v39 = *(_DWORD *)(a2 + 8);
            v40 = *(_DWORD *)(v8 + 8);
            if ( ((v40 & 0x400) != 0) == ((v39 & 0x400) != 0) && ((v40 & 0x200) != 0) == ((v39 & 0x200) != 0) )
              goto LABEL_15;
            break;
          case 0x10u:
            if ( *(_QWORD *)(v8 + 32) == *(_QWORD *)(a2 + 32) )
              goto LABEL_15;
            break;
          default:
            if ( (unsigned int)v20 - 17 > 1
              || (v20 == 18) == (v19 == 18) && *(_DWORD *)(a2 + 32) == *(_DWORD *)(v8 + 32) )
            {
              goto LABEL_15;
            }
            break;
        }
      }
    }
LABEL_2:
    LODWORD(v3) = 0;
    return (unsigned int)v3;
  }
  LODWORD(v3) = (*(_DWORD *)(v8 + 8) & 0x100) == 0;
  if ( (*(_DWORD *)(v8 + 8) & 0x100) != 0 )
  {
    LODWORD(v3) = (*(_DWORD *)(a2 + 8) & 0x100) == 0;
    if ( (*(_DWORD *)(a2 + 8) & 0x100) != 0 )
      goto LABEL_11;
    if ( !*(_BYTE *)(a1 + 500) )
      goto LABEL_58;
    v29 = *(_QWORD **)(a1 + 480);
    v9 = *(unsigned int *)(a1 + 492);
    v14 = (__int64)&v29[v9];
    if ( v29 != (_QWORD *)v14 )
    {
      while ( a2 != *v29 )
      {
        if ( (_QWORD *)v14 == ++v29 )
          goto LABEL_39;
      }
      goto LABEL_2;
    }
LABEL_39:
    if ( (unsigned int)v9 < *(_DWORD *)(a1 + 488) )
    {
      *(_DWORD *)(a1 + 492) = v9 + 1;
      *(_QWORD *)v14 = a2;
      ++*(_QWORD *)(a1 + 472);
    }
    else
    {
LABEL_58:
      sub_C8CC70(a1 + 472, a2, v14, v9, v10, (__int64)v13);
      if ( !v38 )
        goto LABEL_2;
    }
    v30 = *(unsigned int *)(a1 + 336);
    if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 340) )
    {
      sub_C8D5F0(a1 + 328, (const void *)(a1 + 344), v30 + 1, 8u, v10, (__int64)v13);
      v30 = *(unsigned int *)(a1 + 336);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 328) + 8 * v30) = v8;
    v31 = *(unsigned int *)(a1 + 48);
    v32 = *(unsigned int *)(a1 + 52);
    ++*(_DWORD *)(a1 + 336);
    v33 = v41;
    if ( v31 + 1 > v32 )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v31 + 1, 8u, v10, (__int64)v13);
      v31 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v31) = v33;
    v34 = *(unsigned int *)(a1 + 192);
    v35 = *(unsigned int *)(a1 + 196);
    ++*(_DWORD *)(a1 + 48);
    if ( v34 + 1 > v35 )
    {
      sub_C8D5F0(a1 + 184, (const void *)(a1 + 200), v34 + 1, 8u, v10, (__int64)v13);
      v34 = *(unsigned int *)(a1 + 192);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v34) = a2;
    ++*(_DWORD *)(a1 + 192);
    *v17 = a2;
  }
  else
  {
    *v17 = a2;
    v36 = *(unsigned int *)(a1 + 48);
    v37 = v41;
    if ( v36 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
    {
      sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v36 + 1, 8u, v10, (__int64)v13);
      v36 = *(unsigned int *)(a1 + 48);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v36) = v37;
    ++*(_DWORD *)(a1 + 48);
  }
  return (unsigned int)v3;
}

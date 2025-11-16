// Function: sub_1054520
// Address: 0x1054520
//
__int64 __fastcall sub_1054520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  __int64 v8; // r15
  unsigned int v9; // ebx
  int v10; // r11d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 *v13; // r13
  int v14; // eax
  int v15; // eax
  int v17; // eax
  int v18; // eax
  int v19; // r11d
  __int64 v20; // r10
  int v21; // r11d
  __int64 v22; // r10

  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x4000000000LL;
  *(_QWORD *)(a1 + 664) = a1 + 680;
  *(_QWORD *)(a1 + 672) = 0x800000000LL;
  *(_QWORD *)(a1 + 1256) = a1 + 1272;
  *(_QWORD *)a1 = a2;
  *(_DWORD *)(a1 + 8) = a5;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 576) = 0;
  *(_QWORD *)(a1 + 584) = 0;
  *(_QWORD *)(a1 + 592) = 0;
  *(_DWORD *)(a1 + 600) = 0;
  *(_QWORD *)(a1 + 608) = a3;
  *(_QWORD *)(a1 + 616) = a4;
  *(_DWORD *)(a1 + 624) = a4;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)(a1 + 648) = 0;
  *(_DWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 1264) = 0x600000000LL;
  *(_DWORD *)(a1 + 1320) = 0;
  *(_QWORD *)(a1 + 1328) = 0;
  *(_QWORD *)(a1 + 1336) = 0;
  *(_QWORD *)(a1 + 1344) = 0;
  *(_DWORD *)(a1 + 1352) = 0;
  *(_BYTE *)(a1 + 1360) = 0;
  if ( (_DWORD)a4 )
  {
    v7 = a3;
    v8 = a1 + 632;
    a2 = 0;
    a5 = 0;
    v9 = 0;
    while ( 1 )
    {
      v13 = (__int64 *)(v7 + 8LL * v9);
      if ( !(_DWORD)a2 )
        break;
      a4 = *v13;
      v10 = 1;
      v11 = 0;
      a3 = ((_DWORD)a2 - 1) & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
      v12 = a5 + 16 * a3;
      a6 = *(_QWORD *)v12;
      if ( *(_QWORD *)v12 == *v13 )
      {
LABEL_4:
        *(_DWORD *)(v12 + 8) = v9++;
        if ( *(_DWORD *)(a1 + 624) <= v9 )
          return sub_1052400(a1, a2, a3, a4, a5, a6);
        goto LABEL_5;
      }
      while ( a6 != -4096 )
      {
        if ( a6 == -8192 && !v11 )
          v11 = v12;
        a3 = ((_DWORD)a2 - 1) & (unsigned int)(v10 + a3);
        v12 = a5 + 16LL * (unsigned int)a3;
        a6 = *(_QWORD *)v12;
        if ( a4 == *(_QWORD *)v12 )
          goto LABEL_4;
        ++v10;
      }
      if ( !v11 )
        v11 = v12;
      v17 = *(_DWORD *)(a1 + 648);
      ++*(_QWORD *)(a1 + 632);
      v15 = v17 + 1;
      if ( 4 * v15 >= (unsigned int)(3 * a2) )
        goto LABEL_8;
      a5 = (unsigned int)(a2 - (v15 + *(_DWORD *)(a1 + 652)));
      a3 = (unsigned int)a2 >> 3;
      if ( (unsigned int)a5 <= (unsigned int)a3 )
      {
        sub_1051590(v8, a2);
        v18 = *(_DWORD *)(a1 + 656);
        if ( !v18 )
        {
LABEL_45:
          ++*(_DWORD *)(a1 + 648);
          BUG();
        }
        a5 = *v13;
        a2 = (unsigned int)(v18 - 1);
        v19 = 1;
        v20 = 0;
        a6 = *(_QWORD *)(a1 + 640);
        a3 = (unsigned int)a2 & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
        v15 = *(_DWORD *)(a1 + 648) + 1;
        v11 = a6 + 16 * a3;
        a4 = *(_QWORD *)v11;
        if ( *v13 != *(_QWORD *)v11 )
        {
          while ( a4 != -4096 )
          {
            if ( a4 == -8192 && !v20 )
              v20 = v11;
            a3 = (unsigned int)a2 & (v19 + (_DWORD)a3);
            v11 = a6 + 16LL * (unsigned int)a3;
            a4 = *(_QWORD *)v11;
            if ( a5 == *(_QWORD *)v11 )
              goto LABEL_10;
            ++v19;
          }
          a4 = *v13;
          if ( v20 )
            v11 = v20;
        }
      }
LABEL_10:
      *(_DWORD *)(a1 + 648) = v15;
      if ( *(_QWORD *)v11 != -4096 )
        --*(_DWORD *)(a1 + 652);
      *(_QWORD *)v11 = a4;
      *(_DWORD *)(v11 + 8) = 0;
      *(_DWORD *)(v11 + 8) = v9++;
      if ( *(_DWORD *)(a1 + 624) <= v9 )
        return sub_1052400(a1, a2, a3, a4, a5, a6);
LABEL_5:
      a5 = *(_QWORD *)(a1 + 640);
      a2 = *(unsigned int *)(a1 + 656);
    }
    ++*(_QWORD *)(a1 + 632);
LABEL_8:
    sub_1051590(v8, 2 * a2);
    v14 = *(_DWORD *)(a1 + 656);
    if ( !v14 )
      goto LABEL_45;
    a4 = *v13;
    a2 = (unsigned int)(v14 - 1);
    a5 = *(_QWORD *)(a1 + 640);
    a3 = (unsigned int)a2 & (((unsigned int)*v13 >> 9) ^ ((unsigned int)*v13 >> 4));
    v15 = *(_DWORD *)(a1 + 648) + 1;
    v11 = a5 + 16 * a3;
    a6 = *(_QWORD *)v11;
    if ( *(_QWORD *)v11 != *v13 )
    {
      v21 = 1;
      v22 = 0;
      while ( a6 != -4096 )
      {
        if ( !v22 && a6 == -8192 )
          v22 = v11;
        a3 = (unsigned int)a2 & (v21 + (_DWORD)a3);
        v11 = a5 + 16LL * (unsigned int)a3;
        a6 = *(_QWORD *)v11;
        if ( a4 == *(_QWORD *)v11 )
          goto LABEL_10;
        ++v21;
      }
      if ( v22 )
        v11 = v22;
    }
    goto LABEL_10;
  }
  return sub_1052400(a1, a2, a3, a4, a5, a6);
}

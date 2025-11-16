// Function: sub_1626AA0
// Address: 0x1626aa0
//
__int64 __fastcall sub_1626AA0(__int64 a1, int a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 *v10; // rdi
  __int64 v11; // rcx
  int v13; // eax
  int v14; // esi
  __int64 v15; // r9
  __int64 v16; // rax
  __int64 v17; // r8
  int v18; // ecx
  int v19; // r11d
  __int64 *v20; // r10
  int v21; // eax
  int v22; // r11d
  __int64 *v23; // r10
  __int64 v24; // [rsp-38h] [rbp-38h] BYREF
  __int64 *v25; // [rsp-30h] [rbp-30h] BYREF

  if ( (*(_BYTE *)(a1 + 34) & 0x10) != 0 )
  {
    v24 = a1;
    v3 = sub_16498A0(a1);
    v4 = *(_QWORD *)v3;
    v5 = *(_DWORD *)(*(_QWORD *)v3 + 2760LL);
    v6 = *(_QWORD *)v3 + 2736LL;
    if ( v5 )
    {
      v7 = v24;
      v8 = *(_QWORD *)(v4 + 2744);
      v9 = (v5 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v10 = (__int64 *)(v8 + 40 * v9);
      v11 = *v10;
      if ( v24 == *v10 )
        return sub_161F760(v10 + 1, a2);
      v19 = 1;
      v20 = 0;
      while ( v11 != -8 )
      {
        if ( !v20 && v11 == -16 )
          v20 = v10;
        LODWORD(v9) = (v5 - 1) & (v19 + v9);
        v10 = (__int64 *)(v8 + 40LL * (unsigned int)v9);
        v11 = *v10;
        if ( v24 == *v10 )
          return sub_161F760(v10 + 1, a2);
        ++v19;
      }
      v21 = *(_DWORD *)(v4 + 2752);
      if ( v20 )
        v10 = v20;
      ++*(_QWORD *)(v4 + 2736);
      v18 = v21 + 1;
      if ( 4 * (v21 + 1) < 3 * v5 )
      {
        if ( v5 - *(_DWORD *)(v4 + 2756) - v18 <= v5 >> 3 )
        {
          sub_16261B0(v6, v5);
          sub_1621520(v6, &v24, &v25);
          v10 = v25;
          v7 = v24;
          v18 = *(_DWORD *)(v4 + 2752) + 1;
        }
LABEL_9:
        *(_DWORD *)(v4 + 2752) = v18;
        if ( *v10 != -8 )
          --*(_DWORD *)(v4 + 2756);
        *v10 = v7;
        v10[1] = (__int64)(v10 + 3);
        v10[2] = 0x100000000LL;
        return sub_161F760(v10 + 1, a2);
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 2736);
    }
    sub_16261B0(v6, 2 * v5);
    v13 = *(_DWORD *)(v4 + 2760);
    if ( !v13 )
    {
      ++*(_DWORD *)(v4 + 2752);
      BUG();
    }
    v7 = v24;
    v14 = v13 - 1;
    v15 = *(_QWORD *)(v4 + 2744);
    v16 = (v13 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v10 = (__int64 *)(v15 + 40 * v16);
    v17 = *v10;
    v18 = *(_DWORD *)(v4 + 2752) + 1;
    if ( *v10 != v24 )
    {
      v22 = 1;
      v23 = 0;
      while ( v17 != -8 )
      {
        if ( !v23 && v17 == -16 )
          v23 = v10;
        LODWORD(v16) = v14 & (v22 + v16);
        v10 = (__int64 *)(v15 + 40LL * (unsigned int)v16);
        v17 = *v10;
        if ( v24 == *v10 )
          goto LABEL_9;
        ++v22;
      }
      if ( v23 )
        v10 = v23;
    }
    goto LABEL_9;
  }
  return 0;
}

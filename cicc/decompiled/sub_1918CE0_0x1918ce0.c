// Function: sub_1918CE0
// Address: 0x1918ce0
//
void __fastcall sub_1918CE0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 *v5; // r12
  __int64 v7; // r8
  __int64 *v8; // r10
  int v9; // r11d
  unsigned int v10; // eax
  __int64 *v11; // rdi
  __int64 v12; // rcx
  unsigned int v13; // esi
  int v14; // eax
  int v15; // ecx
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 v18; // rdi
  int v19; // edx
  int v20; // r11d
  __int64 *v21; // r9
  int v22; // eax
  __int64 v23; // rax
  _BYTE *v24; // rsi
  int v25; // eax
  int v26; // ecx
  __int64 v27; // r8
  int v28; // r11d
  unsigned int v29; // eax
  __int64 v30; // rdi

  if ( a2 != a3 )
  {
    v4 = a1 + 32;
    v5 = a2;
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
        break;
      v7 = *(_QWORD *)(a1 + 8);
      v8 = 0;
      v9 = 1;
      v10 = (v13 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
      v11 = (__int64 *)(v7 + 8LL * v10);
      v12 = *v11;
      if ( *v11 == *v5 )
      {
LABEL_4:
        if ( a3 == ++v5 )
          return;
      }
      else
      {
        while ( v12 != -8 )
        {
          if ( v12 != -16 || v8 )
            v11 = v8;
          v10 = (v13 - 1) & (v9 + v10);
          v12 = *(_QWORD *)(v7 + 8LL * v10);
          if ( *v5 == v12 )
            goto LABEL_4;
          ++v9;
          v8 = v11;
          v11 = (__int64 *)(v7 + 8LL * v10);
        }
        v22 = *(_DWORD *)(a1 + 16);
        if ( !v8 )
          v8 = v11;
        ++*(_QWORD *)a1;
        v19 = v22 + 1;
        if ( 4 * (v22 + 1) >= 3 * v13 )
          goto LABEL_7;
        if ( v13 - *(_DWORD *)(a1 + 20) - v19 <= v13 >> 3 )
        {
          sub_13B3D40(a1, v13);
          v25 = *(_DWORD *)(a1 + 24);
          if ( !v25 )
          {
LABEL_46:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v26 = v25 - 1;
          v27 = *(_QWORD *)(a1 + 8);
          v21 = 0;
          v28 = 1;
          v29 = (v25 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
          v8 = (__int64 *)(v27 + 8LL * v29);
          v30 = *v8;
          v19 = *(_DWORD *)(a1 + 16) + 1;
          if ( *v5 != *v8 )
          {
            while ( v30 != -8 )
            {
              if ( !v21 && v30 == -16 )
                v21 = v8;
              v29 = v26 & (v28 + v29);
              v8 = (__int64 *)(v27 + 8LL * v29);
              v30 = *v8;
              if ( *v5 == *v8 )
                goto LABEL_23;
              ++v28;
            }
            goto LABEL_11;
          }
        }
LABEL_23:
        *(_DWORD *)(a1 + 16) = v19;
        if ( *v8 != -8 )
          --*(_DWORD *)(a1 + 20);
        v23 = *v5;
        *v8 = *v5;
        v24 = *(_BYTE **)(a1 + 40);
        if ( v24 == *(_BYTE **)(a1 + 48) )
        {
          sub_1292090(v4, v24, v5);
          goto LABEL_4;
        }
        if ( v24 )
        {
          *(_QWORD *)v24 = v23;
          v24 = *(_BYTE **)(a1 + 40);
        }
        ++v5;
        *(_QWORD *)(a1 + 40) = v24 + 8;
        if ( a3 == v5 )
          return;
      }
    }
    ++*(_QWORD *)a1;
LABEL_7:
    sub_13B3D40(a1, 2 * v13);
    v14 = *(_DWORD *)(a1 + 24);
    if ( !v14 )
      goto LABEL_46;
    v15 = v14 - 1;
    v16 = *(_QWORD *)(a1 + 8);
    v17 = (v14 - 1) & (((unsigned int)*v5 >> 9) ^ ((unsigned int)*v5 >> 4));
    v8 = (__int64 *)(v16 + 8LL * v17);
    v18 = *v8;
    v19 = *(_DWORD *)(a1 + 16) + 1;
    if ( *v8 != *v5 )
    {
      v20 = 1;
      v21 = 0;
      while ( v18 != -8 )
      {
        if ( v18 == -16 && !v21 )
          v21 = v8;
        v17 = v15 & (v20 + v17);
        v8 = (__int64 *)(v16 + 8LL * v17);
        v18 = *v8;
        if ( *v5 == *v8 )
          goto LABEL_23;
        ++v20;
      }
LABEL_11:
      if ( v21 )
        v8 = v21;
      goto LABEL_23;
    }
    goto LABEL_23;
  }
}

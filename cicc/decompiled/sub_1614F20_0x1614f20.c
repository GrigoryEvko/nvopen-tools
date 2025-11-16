// Function: sub_1614F20
// Address: 0x1614f20
//
__int64 __fastcall sub_1614F20(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rsi
  __int64 v6; // rcx
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  int v11; // r11d
  _QWORD *v12; // r14
  int v13; // eax
  int v14; // edx
  __int64 v15; // rax
  int v16; // eax
  int v17; // ecx
  unsigned int v18; // eax
  int v19; // r9d
  _QWORD *v20; // r8
  int v21; // eax
  int v22; // eax
  unsigned int v23; // r12d
  int v24; // r8d
  __int64 v25; // rcx

  v4 = a1 + 704;
  v5 = *(unsigned int *)(a1 + 728);
  if ( !(_DWORD)v5 )
  {
    ++*(_QWORD *)(a1 + 704);
    goto LABEL_16;
  }
  v6 = *(_QWORD *)(a1 + 712);
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (_QWORD *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( *v8 != a2 )
  {
    v11 = 1;
    v12 = 0;
    while ( v9 != -4 )
    {
      if ( !v12 && v9 == -8 )
        v12 = v8;
      v7 = (v5 - 1) & (v11 + v7);
      v8 = (_QWORD *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v8;
    v13 = *(_DWORD *)(a1 + 720);
    ++*(_QWORD *)(a1 + 704);
    v14 = v13 + 1;
    if ( 4 * (v13 + 1) < (unsigned int)(3 * v5) )
    {
      if ( (int)v5 - *(_DWORD *)(a1 + 724) - v14 > (unsigned int)v5 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 720) = v14;
        if ( *v12 != -4 )
          --*(_DWORD *)(a1 + 724);
        *v12 = a2;
        v12[1] = 0;
        goto LABEL_14;
      }
      sub_1614D60(v4, v5);
      v21 = *(_DWORD *)(a1 + 728);
      if ( v21 )
      {
        v22 = v21 - 1;
        v5 = *(_QWORD *)(a1 + 712);
        v4 = 0;
        v23 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v24 = 1;
        v14 = *(_DWORD *)(a1 + 720) + 1;
        v12 = (_QWORD *)(v5 + 16LL * v23);
        v25 = *v12;
        if ( *v12 != a2 )
        {
          while ( v25 != -4 )
          {
            if ( !v4 && v25 == -8 )
              v4 = (__int64)v12;
            v23 = v22 & (v24 + v23);
            v12 = (_QWORD *)(v5 + 16LL * v23);
            v25 = *v12;
            if ( *v12 == a2 )
              goto LABEL_11;
            ++v24;
          }
          if ( v4 )
            v12 = (_QWORD *)v4;
        }
        goto LABEL_11;
      }
LABEL_45:
      ++*(_DWORD *)(a1 + 720);
      BUG();
    }
LABEL_16:
    sub_1614D60(v4, 2 * v5);
    v16 = *(_DWORD *)(a1 + 728);
    if ( v16 )
    {
      v17 = v16 - 1;
      v4 = *(_QWORD *)(a1 + 712);
      v18 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = *(_DWORD *)(a1 + 720) + 1;
      v12 = (_QWORD *)(v4 + 16LL * v18);
      v5 = *v12;
      if ( *v12 != a2 )
      {
        v19 = 1;
        v20 = 0;
        while ( v5 != -4 )
        {
          if ( !v20 && v5 == -8 )
            v20 = v12;
          v18 = v17 & (v19 + v18);
          v12 = (_QWORD *)(v4 + 16LL * v18);
          v5 = *v12;
          if ( *v12 == a2 )
            goto LABEL_11;
          ++v19;
        }
        if ( v20 )
          v12 = v20;
      }
      goto LABEL_11;
    }
    goto LABEL_45;
  }
LABEL_3:
  if ( v8[1] )
    return v8[1];
  v12 = v8;
LABEL_14:
  v15 = sub_163A1D0(v4, v5);
  result = sub_163A340(v15, a2);
  v12[1] = result;
  return result;
}

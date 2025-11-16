// Function: sub_B4F660
// Address: 0xb4f660
//
__int64 __fastcall sub_B4F660(__int64 a1, _DWORD *a2, int *a3)
{
  __int64 v3; // r11
  unsigned __int64 v4; // r9
  unsigned __int64 v5; // r8
  __int64 v6; // rsi
  __int64 v7; // rbx
  unsigned __int64 v8; // rcx
  _DWORD *v9; // rax
  __int64 v10; // rdx
  _DWORD *v11; // r10
  __int64 v12; // rcx
  __int64 v13; // rdx
  _DWORD *v14; // rcx
  int v15; // edx
  int v16; // edx
  int v17; // edx
  unsigned __int64 v18; // rcx
  unsigned __int64 v19; // rax
  unsigned __int64 v21; // rsi
  _DWORD *v22; // rdi

  if ( *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL) == 18 )
    return 0;
  v18 = *(int *)(*(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL) + 32LL);
  *a3 = v18;
  v19 = *(unsigned int *)(a1 + 80) / v18;
  if ( *(unsigned int *)(a1 + 80) % v18 )
    return 0;
  *a2 = v19;
  v21 = *(unsigned int *)(a1 + 80);
  v22 = *(_DWORD **)(a1 + 72);
  v3 = *a3;
  if ( *a3 )
  {
    v4 = (int)v19;
    v5 = v21;
    v6 = 0;
    v7 = 4LL * (int)v19;
    while ( 1 )
    {
      v8 = v5;
      v9 = v22;
      if ( v4 <= v5 )
        v8 = v4;
      v10 = 4 * v8;
      v11 = &v22[v8];
      v12 = (__int64)(4 * v8) >> 4;
      v13 = v10 >> 2;
      if ( v12 > 0 )
      {
        v14 = &v22[4 * v12];
        while ( *v9 == -1 || *v9 == (_DWORD)v6 )
        {
          v15 = v9[1];
          if ( v15 != -1 && (_DWORD)v6 != v15 )
          {
            ++v9;
            break;
          }
          v16 = v9[2];
          if ( v16 != -1 && (_DWORD)v6 != v16 )
          {
            v9 += 2;
            break;
          }
          v17 = v9[3];
          if ( v17 != -1 && (_DWORD)v6 != v17 )
          {
            v9 += 3;
            break;
          }
          v9 += 4;
          if ( v14 == v9 )
          {
            v13 = v11 - v9;
            goto LABEL_23;
          }
        }
LABEL_19:
        if ( v11 != v9 )
          return 0;
        goto LABEL_20;
      }
LABEL_23:
      if ( v13 != 2 )
      {
        if ( v13 != 3 )
        {
          if ( v13 != 1 )
            goto LABEL_20;
          goto LABEL_26;
        }
        if ( *v9 != -1 && *v9 != (_DWORD)v6 )
          goto LABEL_19;
        ++v9;
      }
      if ( *v9 != -1 && *v9 != (_DWORD)v6 )
        goto LABEL_19;
      ++v9;
LABEL_26:
      if ( *v9 != -1 && *v9 != (_DWORD)v6 && v11 != v9 )
        return 0;
LABEL_20:
      ++v6;
      v5 -= v4;
      v22 = (_DWORD *)((char *)v22 + v7);
      if ( v3 == v6 )
        return 1;
    }
  }
  return 1;
}

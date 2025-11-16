// Function: sub_2B0C5A0
// Address: 0x2b0c5a0
//
__int64 __fastcall sub_2B0C5A0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r12d
  unsigned int *v3; // rdx
  __int64 v5; // rdi
  int v6; // r10d
  __int64 v7; // rdx
  int v8; // r8d
  int v9; // r9d
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r11
  int v13; // edx
  __int64 v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rcx
  _DWORD *v20; // rax
  _DWORD *v21; // rcx
  _BYTE *v22; // rax
  unsigned int v24; // eax
  __int64 v25; // r14

  v2 = 0;
  if ( *(_BYTE *)(**(_QWORD **)a1 + 8LL * a2 + 4) )
  {
    v3 = *(unsigned int **)(a1 + 16);
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_DWORD *)(v5 + 120);
    v7 = *v3;
    v8 = v6;
    if ( !v6 )
      v8 = *(_DWORD *)(v5 + 8);
    v9 = v7 * a2;
    v10 = v8 - (unsigned int)v7 * a2;
    if ( (unsigned int)v10 > (unsigned int)v7 )
      v10 = v7;
    if ( v10 )
    {
      v2 = 0;
      v11 = 0;
      v12 = **(_QWORD **)(a1 + 24);
      while ( 1 )
      {
        v13 = v9 + v11;
        if ( *(_DWORD *)(v12 + 4LL * (v9 + (int)v11)) == -1 )
          goto LABEL_23;
        if ( v6 )
          v13 = *(_DWORD *)(*(_QWORD *)(v5 + 112) + 4LL * v13);
        if ( v13 == -1 )
          goto LABEL_23;
        v14 = *(unsigned int *)(v5 + 152);
        if ( (_DWORD)v14 )
          break;
LABEL_22:
        v22 = *(_BYTE **)(*(_QWORD *)v5 + 8LL * v13);
        if ( *v22 == 90 )
        {
          v24 = *(_DWORD *)(*(_QWORD *)(*((_QWORD *)v22 - 8) + 8LL) + 32LL);
          if ( v2 < v24 )
            v2 = v24;
          if ( v10 == ++v11 )
            return v2;
        }
        else
        {
LABEL_23:
          if ( v10 == ++v11 )
            return v2;
        }
      }
      v15 = 4 * v14;
      v16 = *(_QWORD *)(v5 + 144);
      v17 = v16 + v15;
      v18 = v15 >> 2;
      v19 = v15 >> 4;
      if ( v19 )
      {
        v20 = *(_DWORD **)(v5 + 144);
        v21 = (_DWORD *)(v16 + 16 * v19);
        while ( *v20 != v13 )
        {
          if ( v20[1] == v13 )
          {
            LODWORD(v18) = ((__int64)v20 - v16 + 4) >> 2;
            goto LABEL_21;
          }
          if ( v20[2] == v13 )
          {
            LODWORD(v18) = ((__int64)v20 - v16 + 8) >> 2;
            goto LABEL_21;
          }
          if ( v20[3] == v13 )
          {
            LODWORD(v18) = ((__int64)v20 - v16 + 12) >> 2;
            goto LABEL_21;
          }
          v20 += 4;
          if ( v21 == v20 )
          {
            v25 = (v17 - (__int64)v20) >> 2;
            goto LABEL_33;
          }
        }
        goto LABEL_20;
      }
      v25 = v18;
      v20 = *(_DWORD **)(v5 + 144);
LABEL_33:
      if ( v25 != 2 )
      {
        if ( v25 != 3 )
        {
          if ( v25 != 1 || *v20 != v13 )
            goto LABEL_21;
          goto LABEL_20;
        }
        if ( *v20 == v13 )
        {
LABEL_20:
          LODWORD(v18) = ((__int64)v20 - v16) >> 2;
LABEL_21:
          v13 = v18;
          goto LABEL_22;
        }
        ++v20;
      }
      if ( *v20 != v13 && *++v20 != v13 )
        goto LABEL_21;
      goto LABEL_20;
    }
    return 0;
  }
  return v2;
}

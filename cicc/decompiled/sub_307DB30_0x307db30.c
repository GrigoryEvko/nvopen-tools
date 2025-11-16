// Function: sub_307DB30
// Address: 0x307db30
//
__int64 __fastcall sub_307DB30(__int64 a1, int a2, __int64 a3)
{
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // r9d
  int v8; // r11d
  int v9; // r13d
  unsigned int v10; // r8d
  __int64 v11; // r15
  _DWORD *v12; // rax
  int v13; // edi
  int v14; // r10d
  unsigned int v15; // eax
  char v16; // cl
  __int64 v17; // rax
  int v19; // r11d
  _DWORD *v20; // r10
  int v21; // eax
  int v22; // ecx
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  unsigned int v26; // r13d
  int v27; // esi
  int v28; // r9d
  _DWORD *v29; // r8
  int v30; // eax
  int v31; // eax
  __int64 v32; // rdi
  unsigned int v33; // r13d
  int v34; // r9d
  int v35; // esi
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 80);
  v5 = *(_QWORD *)(a1 + 64);
  if ( v4 )
  {
    v6 = v4 - 1;
    v8 = 1;
    v9 = 37 * a2;
    v10 = (v4 - 1) & (37 * a2);
    LODWORD(v11) = v10;
    v12 = (_DWORD *)(v5 + 8LL * v10);
    v13 = *v12;
    v14 = *v12;
    if ( *v12 == a2 )
    {
LABEL_3:
      v15 = v12[1];
      v16 = v15 & 0x3F;
      v17 = 8LL * (v15 >> 6);
      return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v17) >> v16) & 1LL;
    }
    while ( 1 )
    {
      if ( v14 == -1 )
        return 0;
      v11 = v6 & ((_DWORD)v11 + v8);
      v14 = *(_DWORD *)(v5 + 8 * v11);
      if ( v14 == a2 )
        break;
      ++v8;
    }
    v19 = 1;
    v20 = 0;
    while ( v13 != -1 )
    {
      if ( !v20 && v13 == -2 )
        v20 = v12;
      v10 = v6 & (v19 + v10);
      v12 = (_DWORD *)(v5 + 8LL * v10);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_3;
      ++v19;
    }
    if ( !v20 )
      v20 = v12;
    v21 = *(_DWORD *)(a1 + 72);
    ++*(_QWORD *)(a1 + 56);
    v22 = v21 + 1;
    if ( 4 * (v21 + 1) >= 3 * v4 )
    {
      v36 = a3;
      sub_2E518D0(a1 + 56, 2 * v4);
      v23 = *(_DWORD *)(a1 + 80);
      if ( v23 )
      {
        v24 = v23 - 1;
        v25 = *(_QWORD *)(a1 + 64);
        a3 = v36;
        v26 = v24 & v9;
        v20 = (_DWORD *)(v25 + 8LL * v26);
        v27 = *v20;
        v22 = *(_DWORD *)(a1 + 72) + 1;
        if ( *v20 == a2 )
          goto LABEL_15;
        v28 = 1;
        v29 = 0;
        while ( v27 != -1 )
        {
          if ( v27 == -2 && !v29 )
            v29 = v20;
          v26 = v24 & (v28 + v26);
          v20 = (_DWORD *)(v25 + 8LL * v26);
          v27 = *v20;
          if ( *v20 == a2 )
            goto LABEL_15;
          ++v28;
        }
LABEL_22:
        if ( v29 )
          v20 = v29;
        goto LABEL_15;
      }
    }
    else
    {
      if ( v4 - *(_DWORD *)(a1 + 76) - v22 > v4 >> 3 )
      {
LABEL_15:
        *(_DWORD *)(a1 + 72) = v22;
        if ( *v20 != -1 )
          --*(_DWORD *)(a1 + 76);
        *v20 = a2;
        v16 = 0;
        v17 = 0;
        v20[1] = 0;
        return (*(_QWORD *)(*(_QWORD *)(a3 + 24) + v17) >> v16) & 1LL;
      }
      v37 = a3;
      sub_2E518D0(a1 + 56, v4);
      v30 = *(_DWORD *)(a1 + 80);
      if ( v30 )
      {
        v31 = v30 - 1;
        v32 = *(_QWORD *)(a1 + 64);
        v29 = 0;
        a3 = v37;
        v33 = v31 & v9;
        v34 = 1;
        v20 = (_DWORD *)(v32 + 8LL * v33);
        v35 = *v20;
        v22 = *(_DWORD *)(a1 + 72) + 1;
        if ( *v20 == a2 )
          goto LABEL_15;
        while ( v35 != -1 )
        {
          if ( !v29 && v35 == -2 )
            v29 = v20;
          v33 = v31 & (v34 + v33);
          v20 = (_DWORD *)(v32 + 8LL * v33);
          v35 = *v20;
          if ( *v20 == a2 )
            goto LABEL_15;
          ++v34;
        }
        goto LABEL_22;
      }
    }
    ++*(_DWORD *)(a1 + 72);
    BUG();
  }
  return 0;
}

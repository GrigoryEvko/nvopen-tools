// Function: sub_391FDB0
// Address: 0x391fdb0
//
void __fastcall sub_391FDB0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // r9
  unsigned int v7; // edx
  _QWORD *v8; // rax
  __int64 v9; // r8
  __int64 v10; // r8
  unsigned __int64 v11; // r9
  unsigned int v12; // esi
  _BYTE *v13; // rdx
  unsigned int v14; // esi
  int v15; // esi
  __int64 v16; // rdi
  int v17; // esi
  __int64 v18; // r9
  unsigned int v19; // edx
  int v20; // eax
  _QWORD *v21; // rcx
  __int64 v22; // r8
  __int64 v23; // rax
  int v24; // r11d
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  int v28; // edx
  int v29; // r11d
  _QWORD *v30; // r10
  __int64 v31; // r9
  unsigned int v32; // esi
  __int64 v33; // r8
  int v34; // ecx
  __int64 *v35; // r11
  int v36; // r11d

  v1 = *(_QWORD *)(a1 + 224);
  v2 = *(_QWORD *)(a1 + 232);
  if ( v1 != v2 )
  {
    v4 = a1 + 248;
    do
    {
      v14 = *(_DWORD *)(a1 + 272);
      if ( v14 )
      {
        v5 = *(_QWORD *)(v1 + 16);
        v6 = *(_QWORD *)(a1 + 256);
        v7 = (v14 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v8 = (_QWORD *)(v6 + 32LL * v7);
        v9 = *v8;
        if ( *v8 == v5 )
        {
          v10 = v8[1];
          v11 = 0xCCCCCCCCCCCCCCCDLL * ((v8[2] - v10) >> 3);
          goto LABEL_5;
        }
        v24 = 1;
        v21 = 0;
        while ( v9 != -8 )
        {
          if ( v21 || v9 != -16 )
            v8 = v21;
          v34 = v24 + 1;
          v7 = (v14 - 1) & (v24 + v7);
          v35 = (__int64 *)(v6 + 32LL * v7);
          v9 = *v35;
          if ( v5 == *v35 )
          {
            v10 = v35[1];
            v11 = 0xCCCCCCCCCCCCCCCDLL * ((v35[2] - v10) >> 3);
            goto LABEL_5;
          }
          v24 = v34;
          v21 = v8;
          v8 = (_QWORD *)(v6 + 32LL * v7);
        }
        if ( !v21 )
          v21 = v8;
        v25 = *(_DWORD *)(a1 + 264);
        ++*(_QWORD *)(a1 + 248);
        v20 = v25 + 1;
        if ( 4 * v20 < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 268) - v20 > v14 >> 3 )
            goto LABEL_10;
          sub_391A270(v4, v14);
          v26 = *(_DWORD *)(a1 + 272);
          if ( !v26 )
          {
LABEL_44:
            ++*(_DWORD *)(a1 + 264);
            BUG();
          }
          v27 = *(_QWORD *)(v1 + 16);
          v28 = v26 - 1;
          v29 = 1;
          v30 = 0;
          v31 = *(_QWORD *)(a1 + 256);
          v32 = v28 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
          v20 = *(_DWORD *)(a1 + 264) + 1;
          v21 = (_QWORD *)(v31 + 32LL * v32);
          v33 = *v21;
          if ( *v21 == v27 )
            goto LABEL_10;
          while ( v33 != -8 )
          {
            if ( !v30 && v33 == -16 )
              v30 = v21;
            v32 = v28 & (v29 + v32);
            v21 = (_QWORD *)(v31 + 32LL * v32);
            v33 = *v21;
            if ( v27 == *v21 )
              goto LABEL_10;
            ++v29;
          }
          goto LABEL_23;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 248);
      }
      sub_391A270(v4, 2 * v14);
      v15 = *(_DWORD *)(a1 + 272);
      if ( !v15 )
        goto LABEL_44;
      v16 = *(_QWORD *)(v1 + 16);
      v17 = v15 - 1;
      v18 = *(_QWORD *)(a1 + 256);
      v19 = v17 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v20 = *(_DWORD *)(a1 + 264) + 1;
      v21 = (_QWORD *)(v18 + 32LL * v19);
      v22 = *v21;
      if ( *v21 == v16 )
        goto LABEL_10;
      v36 = 1;
      v30 = 0;
      while ( v22 != -8 )
      {
        if ( !v30 && v22 == -16 )
          v30 = v21;
        v19 = v17 & (v36 + v19);
        v21 = (_QWORD *)(v18 + 32LL * v19);
        v22 = *v21;
        if ( v16 == *v21 )
          goto LABEL_10;
        ++v36;
      }
LABEL_23:
      if ( v30 )
        v21 = v30;
LABEL_10:
      *(_DWORD *)(a1 + 264) = v20;
      if ( *v21 != -8 )
        --*(_DWORD *)(a1 + 268);
      v23 = *(_QWORD *)(v1 + 16);
      v11 = 0;
      v10 = 0;
      v21[1] = 0;
      v21[2] = 0;
      *v21 = v23;
      v21[3] = 0;
LABEL_5:
      v12 = *(_DWORD *)(v1 + 28);
      v13 = *(_BYTE **)v1;
      v1 += 32;
      sub_391F970(a1, v12, v13, *(_QWORD *)(v1 - 24), v10, v11);
    }
    while ( v2 != v1 );
  }
}

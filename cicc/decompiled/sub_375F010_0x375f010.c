// Function: sub_375F010
// Address: 0x375f010
//
void __fastcall sub_375F010(__int64 a1, unsigned __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5)
{
  int v8; // r12d
  char v9; // dl
  __int64 v10; // r9
  int v11; // esi
  unsigned int v12; // ecx
  _DWORD *v13; // rax
  int v14; // r8d
  _DWORD *v15; // r12
  unsigned int v16; // esi
  unsigned int v17; // r8d
  _DWORD *v18; // rdi
  int v19; // eax
  unsigned int v20; // ecx
  int v21; // r10d
  __int64 v22; // rcx
  int v23; // eax
  unsigned int v24; // edx
  int v25; // esi
  __int64 v26; // rcx
  int v27; // edx
  unsigned int v28; // eax
  int v29; // esi
  int v30; // r9d
  _DWORD *v31; // r8
  int v32; // eax
  int v33; // edx
  int v34; // r9d
  unsigned __int64 v35; // [rsp+0h] [rbp-30h] BYREF
  __int64 v36; // [rsp+8h] [rbp-28h]

  v35 = a4;
  v36 = a5;
  sub_375EAB0(a1, (__int64)&v35);
  v8 = sub_375D5B0(a1, a2, a3);
  v9 = *(_BYTE *)(a1 + 720) & 1;
  if ( v9 )
  {
    v10 = a1 + 728;
    v11 = 7;
  }
  else
  {
    v16 = *(_DWORD *)(a1 + 736);
    v10 = *(_QWORD *)(a1 + 728);
    if ( !v16 )
    {
      v17 = *(_DWORD *)(a1 + 720);
      ++*(_QWORD *)(a1 + 712);
      v18 = 0;
      v19 = (v17 >> 1) + 1;
LABEL_9:
      v20 = 3 * v16;
      goto LABEL_10;
    }
    v11 = v16 - 1;
  }
  v12 = v11 & (37 * v8);
  v13 = (_DWORD *)(v10 + 8LL * v12);
  v14 = *v13;
  if ( v8 == *v13 )
  {
LABEL_4:
    v15 = v13 + 1;
    goto LABEL_5;
  }
  v21 = 1;
  v18 = 0;
  while ( v14 != -1 )
  {
    if ( !v18 && v14 == -2 )
      v18 = v13;
    v12 = v11 & (v21 + v12);
    v13 = (_DWORD *)(v10 + 8LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
      goto LABEL_4;
    ++v21;
  }
  v17 = *(_DWORD *)(a1 + 720);
  v20 = 24;
  v16 = 8;
  if ( !v18 )
    v18 = v13;
  ++*(_QWORD *)(a1 + 712);
  v19 = (v17 >> 1) + 1;
  if ( !v9 )
  {
    v16 = *(_DWORD *)(a1 + 736);
    goto LABEL_9;
  }
LABEL_10:
  if ( 4 * v19 >= v20 )
  {
    sub_375BDE0(a1 + 712, 2 * v16);
    if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
    {
      v22 = a1 + 728;
      v23 = 7;
    }
    else
    {
      v32 = *(_DWORD *)(a1 + 736);
      v22 = *(_QWORD *)(a1 + 728);
      if ( !v32 )
        goto LABEL_53;
      v23 = v32 - 1;
    }
    v24 = v23 & (37 * v8);
    v18 = (_DWORD *)(v22 + 8LL * v24);
    v25 = *v18;
    if ( v8 != *v18 )
    {
      v34 = 1;
      v31 = 0;
      while ( v25 != -1 )
      {
        if ( !v31 && v25 == -2 )
          v31 = v18;
        v24 = v23 & (v34 + v24);
        v18 = (_DWORD *)(v22 + 8LL * v24);
        v25 = *v18;
        if ( v8 == *v18 )
          goto LABEL_24;
        ++v34;
      }
      goto LABEL_30;
    }
LABEL_24:
    v17 = *(_DWORD *)(a1 + 720);
    goto LABEL_12;
  }
  if ( v16 - *(_DWORD *)(a1 + 724) - v19 <= v16 >> 3 )
  {
    sub_375BDE0(a1 + 712, v16);
    if ( (*(_BYTE *)(a1 + 720) & 1) != 0 )
    {
      v26 = a1 + 728;
      v27 = 7;
      goto LABEL_27;
    }
    v33 = *(_DWORD *)(a1 + 736);
    v26 = *(_QWORD *)(a1 + 728);
    if ( v33 )
    {
      v27 = v33 - 1;
LABEL_27:
      v28 = v27 & (37 * v8);
      v18 = (_DWORD *)(v26 + 8LL * v28);
      v29 = *v18;
      if ( v8 != *v18 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -1 )
        {
          if ( v29 == -2 && !v31 )
            v31 = v18;
          v28 = v27 & (v30 + v28);
          v18 = (_DWORD *)(v26 + 8LL * v28);
          v29 = *v18;
          if ( v8 == *v18 )
            goto LABEL_24;
          ++v30;
        }
LABEL_30:
        if ( v31 )
          v18 = v31;
        goto LABEL_24;
      }
      goto LABEL_24;
    }
LABEL_53:
    *(_DWORD *)(a1 + 720) = (2 * (*(_DWORD *)(a1 + 720) >> 1) + 2) | *(_DWORD *)(a1 + 720) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 720) = (2 * (v17 >> 1) + 2) | v17 & 1;
  if ( *v18 != -1 )
    --*(_DWORD *)(a1 + 724);
  *v18 = v8;
  v15 = v18 + 1;
  v18[1] = 0;
LABEL_5:
  *v15 = sub_375D5B0(a1, v35, v36);
  sub_33F9B80(*(_QWORD *)(a1 + 8), a2, a3, v35, v36, 0, 0, 1);
}

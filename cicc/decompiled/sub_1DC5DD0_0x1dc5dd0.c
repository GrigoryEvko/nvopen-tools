// Function: sub_1DC5DD0
// Address: 0x1dc5dd0
//
void __fastcall sub_1DC5DD0(__int64 a1, __int64 a2, signed int a3, int a4, __int64 a5)
{
  _QWORD *v9; // rax
  __int64 (*v10)(void); // rdx
  __int64 v11; // rbx
  char v12; // dl
  char v13; // al
  char v14; // cl
  int v15; // eax
  unsigned __int16 v16; // ax
  int v17; // eax
  unsigned __int64 v18; // rdi
  __int64 v19; // r8
  unsigned __int64 v20; // rax
  unsigned int v21; // r9d
  __int64 v22; // rsi
  unsigned __int64 i; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  unsigned int v26; // r10d
  __int64 *v27; // rdx
  __int64 v28; // r8
  unsigned __int64 v29; // rdx
  unsigned __int16 v30; // ax
  int v31; // edx
  __int64 v32; // rax
  int v33; // r11d
  __int64 v34; // [rsp+10h] [rbp-70h]
  __int64 *v35; // [rsp+20h] [rbp-60h] BYREF
  __int64 v36; // [rsp+28h] [rbp-58h]
  _BYTE v37[80]; // [rsp+30h] [rbp-50h] BYREF

  v35 = (__int64 *)v37;
  v36 = 0x400000000LL;
  if ( a5 )
    sub_1DB4D80(a5, (__int64)&v35, a4, *(_QWORD **)(a1 + 8), *(_QWORD *)(a1 + 16));
  v9 = *(_QWORD **)(a1 + 8);
  v34 = 0;
  v10 = *(__int64 (**)(void))(**(_QWORD **)(*v9 + 16LL) + 112LL);
  if ( v10 != sub_1D00B10 )
  {
    v34 = v10();
    v9 = *(_QWORD **)(a1 + 8);
  }
  if ( a3 < 0 )
    v11 = *(_QWORD *)(v9[3] + 16LL * (a3 & 0x7FFFFFFF) + 8);
  else
    v11 = *(_QWORD *)(v9[34] + 8LL * (unsigned int)a3);
  if ( v11 )
  {
    v12 = *(_BYTE *)(v11 + 4);
    if ( (v12 & 8) == 0 )
      goto LABEL_9;
    do
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        goto LABEL_30;
      v12 = *(_BYTE *)(v11 + 4);
    }
    while ( (v12 & 8) != 0 );
LABEL_9:
    v13 = *(_BYTE *)(v11 + 3);
    if ( (v13 & 0x10) == 0 )
      *(_BYTE *)(v11 + 3) = v13 & 0xBF;
    v14 = v12 & 1;
    if ( (v12 & 1) == 0 && (v12 & 2) == 0 )
    {
      v15 = *(_DWORD *)v11 >> 8;
      if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
      {
        v16 = v15 & 0xFFF;
        if ( !v16 || a4 != -1 )
          goto LABEL_29;
        v17 = ~*(_DWORD *)(*(_QWORD *)(v34 + 248) + 4LL * v16);
      }
      else
      {
        v30 = v15 & 0xFFF;
        if ( !v30 )
        {
          v18 = *(_QWORD *)(v11 + 16);
          v19 = *(_QWORD *)(v18 + 32);
          v20 = 0xCCCCCCCCCCCCCCCDLL * ((v11 - v19) >> 3);
          v21 = -858993459 * ((v11 - v19) >> 3);
          if ( **(_WORD **)(v18 + 16) != 45 && **(_WORD **)(v18 + 16) )
            goto LABEL_45;
          goto LABEL_39;
        }
        v17 = *(_DWORD *)(*(_QWORD *)(v34 + 248) + 4LL * v30);
      }
      if ( (a4 & v17) == 0 )
        goto LABEL_29;
      v18 = *(_QWORD *)(v11 + 16);
      v19 = *(_QWORD *)(v18 + 32);
      v20 = 0xCCCCCCCCCCCCCCCDLL * ((v11 - v19) >> 3);
      v21 = -858993459 * ((v11 - v19) >> 3);
      if ( **(_WORD **)(v18 + 16) != 45 && **(_WORD **)(v18 + 16) )
      {
        if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
        {
          v14 = (v12 & 4) != 0;
          goto LABEL_22;
        }
LABEL_45:
        v32 = v19 + 40LL * (unsigned int)v20;
        if ( !*(_BYTE *)v32 && (*(_BYTE *)(v32 + 3) & 0x10) == 0 && (*(_WORD *)(v32 + 2) & 0xFF0) != 0 )
          v14 = (*(_BYTE *)(*(_QWORD *)(v18 + 32) + 40LL * (unsigned int)sub_1E16AB0(v18, v21) + 4) & 4) != 0;
LABEL_22:
        v22 = *(_QWORD *)(a1 + 16);
        for ( i = v18; (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        v24 = *(_QWORD *)(v22 + 368);
        v25 = *(unsigned int *)(v22 + 384);
        if ( (_DWORD)v25 )
        {
          v26 = (v25 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v27 = (__int64 *)(v24 + 16LL * v26);
          v28 = *v27;
          if ( *v27 == i )
          {
LABEL_26:
            v29 = (v14 == 0 ? 4LL : 2LL) | v27[1] & 0xFFFFFFFFFFFFFFF8LL;
            goto LABEL_27;
          }
          v31 = 1;
          while ( v28 != -8 )
          {
            v33 = v31 + 1;
            v26 = (v25 - 1) & (v31 + v26);
            v27 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v27;
            if ( *v27 == i )
              goto LABEL_26;
            v31 = v33;
          }
        }
        v27 = (__int64 *)(v24 + 16 * v25);
        goto LABEL_26;
      }
LABEL_39:
      v29 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 392LL)
                      + 16LL * *(unsigned int *)(*(_QWORD *)(v19 + 40LL * (v21 + 1) + 24) + 48LL)
                      + 8);
LABEL_27:
      sub_1DC5C40((_QWORD *)a1, a2, v29, a3, v35, (unsigned int)v36);
      v11 = *(_QWORD *)(v11 + 32);
      if ( v11 )
        goto LABEL_28;
      goto LABEL_30;
    }
LABEL_29:
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      if ( !v11 )
        break;
LABEL_28:
      v12 = *(_BYTE *)(v11 + 4);
      if ( (v12 & 8) == 0 )
        goto LABEL_9;
    }
  }
LABEL_30:
  if ( v35 != (__int64 *)v37 )
    _libc_free((unsigned __int64)v35);
}

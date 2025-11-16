// Function: sub_1D04AE0
// Address: 0x1d04ae0
//
void __fastcall sub_1D04AE0(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  __int64 v4; // r12
  __int64 v5; // r15
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned int v10; // edx
  __int64 *v11; // r14
  __int64 v12; // r10
  _DWORD *v13; // rax
  __int64 v14; // rdx
  _DWORD *v15; // rdi
  __int64 v16; // r9
  __int64 v17; // rdx
  _DWORD *v18; // rdx
  char v19; // al
  __int64 v20; // rdi
  void (*v21)(void); // rax
  int v22; // eax
  _BYTE *v23; // r9
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  int v26; // r14d
  int v27; // r11d
  unsigned int v28; // [rsp-4Ch] [rbp-4Ch]
  unsigned int v29; // [rsp-4Ch] [rbp-4Ch]
  __int64 v30; // [rsp-40h] [rbp-40h] BYREF

  v2 = *(_DWORD *)(a1 + 752);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = 8 * v4;
    while ( 1 )
    {
      v7 = *(_QWORD *)(a1 + 800);
      v8 = *(_QWORD *)(*(_QWORD *)(a1 + 744) + v5);
      v9 = *(unsigned int *)(a1 + 816);
      if ( (_DWORD)v9 )
      {
        v10 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v11 = (__int64 *)(v7 + 40LL * v10);
        v12 = *v11;
        if ( v8 == *v11 )
          goto LABEL_5;
        v26 = 1;
        while ( v12 != -8 )
        {
          v27 = v26 + 1;
          v10 = (v9 - 1) & (v26 + v10);
          v11 = (__int64 *)(v7 + 40LL * v10);
          v12 = *v11;
          if ( v8 == *v11 )
            goto LABEL_5;
          v26 = v27;
        }
      }
      v11 = (__int64 *)(v7 + 40 * v9);
LABEL_5:
      if ( a2 )
      {
        v13 = (_DWORD *)v11[1];
        v14 = 4LL * *((unsigned int *)v11 + 4);
        v15 = &v13[(unsigned __int64)v14 / 4];
        v16 = v14 >> 2;
        v17 = v14 >> 4;
        if ( v17 )
        {
          v18 = &v13[4 * v17];
          while ( a2 != *v13 )
          {
            if ( a2 == v13[1] )
            {
              ++v13;
              goto LABEL_13;
            }
            if ( a2 == v13[2] )
            {
              v13 += 2;
              goto LABEL_13;
            }
            if ( a2 == v13[3] )
            {
              v13 += 3;
              goto LABEL_13;
            }
            v13 += 4;
            if ( v18 == v13 )
            {
              v16 = v15 - v13;
              goto LABEL_33;
            }
          }
          goto LABEL_13;
        }
LABEL_33:
        if ( v16 != 2 )
        {
          if ( v16 != 3 )
          {
            if ( v16 != 1 || a2 != *v13 )
              goto LABEL_26;
            goto LABEL_13;
          }
          if ( a2 == *v13 )
          {
LABEL_13:
            if ( v15 == v13 )
              goto LABEL_26;
            goto LABEL_14;
          }
          ++v13;
        }
        if ( a2 != *v13 && a2 != *++v13 )
          goto LABEL_26;
        goto LABEL_13;
      }
LABEL_14:
      v19 = *(_BYTE *)(v8 + 229) & 0xFE;
      *(_BYTE *)(v8 + 229) = v19;
      if ( (v19 & 2) != 0 && !*(_DWORD *)(v8 + 196) )
      {
        v20 = *(_QWORD *)(a1 + 672);
        v21 = *(void (**)(void))(*(_QWORD *)v20 + 88LL);
        if ( (char *)v21 == (char *)sub_1D047D0 )
        {
          v22 = *(_DWORD *)(v20 + 40);
          v30 = v8;
          *(_DWORD *)(v20 + 40) = ++v22;
          *(_DWORD *)(v8 + 196) = v22;
          v23 = *(_BYTE **)(v20 + 24);
          if ( v23 == *(_BYTE **)(v20 + 32) )
          {
            v29 = v2;
            sub_1CFD630(v20 + 16, v23, &v30);
            v2 = v29;
          }
          else
          {
            if ( v23 )
            {
              *(_QWORD *)v23 = v8;
              v23 = *(_BYTE **)(v20 + 24);
            }
            *(_QWORD *)(v20 + 24) = v23 + 8;
          }
        }
        else
        {
          v28 = v2;
          v21();
          v2 = v28;
        }
      }
      v24 = *(unsigned int *)(a1 + 752);
      if ( (unsigned int)v24 > v2 )
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 744) + v5) = *(_QWORD *)(*(_QWORD *)(a1 + 744) + 8 * v24 - 8);
        LODWORD(v24) = *(_DWORD *)(a1 + 752);
      }
      *(_DWORD *)(a1 + 752) = v24 - 1;
      v25 = v11[1];
      if ( (__int64 *)v25 != v11 + 3 )
        _libc_free(v25);
      *v11 = -16;
      --*(_DWORD *)(a1 + 808);
      ++*(_DWORD *)(a1 + 812);
LABEL_26:
      v5 -= 8;
      v2 = v4;
      if ( v5 == -8 )
        return;
      LODWORD(v4) = v4 - 1;
    }
  }
}

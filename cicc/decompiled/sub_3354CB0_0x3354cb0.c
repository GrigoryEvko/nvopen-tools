// Function: sub_3354CB0
// Address: 0x3354cb0
//
void __fastcall sub_3354CB0(__int64 a1, int a2)
{
  unsigned int v2; // r8d
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v7; // edx
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int v10; // eax
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

  v2 = *(_DWORD *)(a1 + 720);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = 8 * v4;
    while ( 1 )
    {
      v7 = *(_DWORD *)(a1 + 784);
      v8 = *(_QWORD *)(a1 + 768);
      v9 = *(_QWORD *)(*(_QWORD *)(a1 + 712) + v5);
      if ( v7 )
      {
        v10 = (v7 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v11 = (__int64 *)(v8 + 40LL * v10);
        v12 = *v11;
        if ( v9 == *v11 )
          goto LABEL_5;
        v26 = 1;
        while ( v12 != -4096 )
        {
          v27 = v26 + 1;
          v10 = (v7 - 1) & (v26 + v10);
          v11 = (__int64 *)(v8 + 40LL * v10);
          v12 = *v11;
          if ( v9 == *v11 )
            goto LABEL_5;
          v26 = v27;
        }
      }
      v11 = (__int64 *)(v8 + 40LL * v7);
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
      v19 = *(_BYTE *)(v9 + 249) & 0xFE;
      *(_BYTE *)(v9 + 249) = v19;
      if ( (v19 & 2) != 0 && !*(_DWORD *)(v9 + 204) )
      {
        v20 = *(_QWORD *)(a1 + 640);
        v21 = *(void (**)(void))(*(_QWORD *)v20 + 88LL);
        if ( (char *)v21 == (char *)sub_33549A0 )
        {
          v22 = *(_DWORD *)(v20 + 40);
          v30 = v9;
          *(_DWORD *)(v20 + 40) = ++v22;
          *(_DWORD *)(v9 + 204) = v22;
          v23 = *(_BYTE **)(v20 + 24);
          if ( v23 == *(_BYTE **)(v20 + 32) )
          {
            v29 = v2;
            sub_2ECAD30(v20 + 16, v23, &v30);
            v2 = v29;
          }
          else
          {
            if ( v23 )
            {
              *(_QWORD *)v23 = v9;
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
      v24 = *(unsigned int *)(a1 + 720);
      if ( (unsigned int)v24 > v2 )
      {
        *(_QWORD *)(*(_QWORD *)(a1 + 712) + v5) = *(_QWORD *)(*(_QWORD *)(a1 + 712) + 8 * v24 - 8);
        LODWORD(v24) = *(_DWORD *)(a1 + 720);
      }
      *(_DWORD *)(a1 + 720) = v24 - 1;
      v25 = v11[1];
      if ( (__int64 *)v25 != v11 + 3 )
        _libc_free(v25);
      *v11 = -8192;
      --*(_DWORD *)(a1 + 776);
      ++*(_DWORD *)(a1 + 780);
LABEL_26:
      v5 -= 8;
      v2 = v4;
      if ( v5 == -8 )
        return;
      LODWORD(v4) = v4 - 1;
    }
  }
}

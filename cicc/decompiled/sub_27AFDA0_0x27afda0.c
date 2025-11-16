// Function: sub_27AFDA0
// Address: 0x27afda0
//
void __fastcall sub_27AFDA0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  char *v4; // r13
  _QWORD *v6; // rdi
  _QWORD *v7; // rsi
  bool v8; // al
  __int64 v9; // r8
  char *v10; // r15
  int v11; // edx
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 *v14; // rdi
  int v15; // edx
  unsigned int v16; // eax
  __int64 *v17; // rsi
  __int64 v18; // r10
  int v19; // eax
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  _QWORD *v22; // rax
  int v23; // r8d
  __int64 v24; // rax
  _BYTE *v25; // rdx
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rsi
  int v29; // edx
  unsigned int v30; // eax
  __int64 *v31; // rcx
  __int64 v32; // rdi
  __int64 v33; // rax
  _QWORD *v34; // rdi
  __int64 v35; // rsi
  _QWORD *v36; // rax
  int v37; // r8d
  int v38; // esi
  int v39; // ecx
  int v40; // r10d
  int v41; // r11d
  __int64 v42[8]; // [rsp-40h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a1 + 96);
  if ( *(_DWORD *)(a1 + 104) )
  {
    v4 = *(char **)(a1 + 96);
    while ( 1 )
    {
      v11 = *(_DWORD *)(a2 + 16);
      v9 = *(_QWORD *)(*(_QWORD *)v4 + 40LL);
      v42[0] = v9;
      if ( !v11 )
        break;
      v12 = *(_QWORD *)(a2 + 8);
      v13 = *(unsigned int *)(a2 + 24);
      v14 = (__int64 *)(v12 + 8 * v13);
      if ( !(_DWORD)v13 )
      {
        v10 = v4 + 8;
        goto LABEL_10;
      }
      v15 = v13 - 1;
      v16 = v15 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v17 = (__int64 *)(v12 + 8LL * v16);
      v18 = *v17;
      if ( v9 == *v17 )
      {
LABEL_9:
        v10 = v4 + 8;
        if ( v14 == v17 )
          goto LABEL_10;
LABEL_5:
        v4 = v10;
        if ( v10 == (char *)(v2 + 8LL * *(unsigned int *)(a1 + 104)) )
          return;
      }
      else
      {
        v38 = 1;
        while ( v18 != -4096 )
        {
          v41 = v38 + 1;
          v16 = v15 & (v38 + v16);
          v17 = (__int64 *)(v12 + 8LL * v16);
          v18 = *v17;
          if ( v9 == *v17 )
            goto LABEL_9;
          v38 = v41;
        }
        v8 = 0;
LABEL_4:
        v10 = v4 + 8;
        if ( v8 )
          goto LABEL_5;
LABEL_10:
        v19 = *(_DWORD *)(a1 + 16);
        v42[0] = v9;
        if ( v19 )
        {
          v27 = *(_DWORD *)(a1 + 24);
          v28 = *(_QWORD *)(a1 + 8);
          if ( v27 )
          {
            v29 = v27 - 1;
            v30 = (v27 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v31 = (__int64 *)(v28 + 8LL * v30);
            v32 = *v31;
            if ( v9 == *v31 )
            {
LABEL_21:
              *v31 = -8192;
              v33 = *(unsigned int *)(a1 + 40);
              --*(_DWORD *)(a1 + 16);
              v34 = *(_QWORD **)(a1 + 32);
              ++*(_DWORD *)(a1 + 20);
              v35 = (__int64)&v34[v33];
              v36 = sub_27ABED0(v34, v35, v42);
              if ( v36 + 1 != (_QWORD *)v35 )
              {
                memmove(v36, v36 + 1, v35 - (_QWORD)(v36 + 1));
                v37 = *(_DWORD *)(a1 + 40);
              }
              v2 = *(_QWORD *)(a1 + 96);
              *(_DWORD *)(a1 + 40) = v37 - 1;
            }
            else
            {
              v39 = 1;
              while ( v32 != -4096 )
              {
                v40 = v39 + 1;
                v30 = v29 & (v39 + v30);
                v31 = (__int64 *)(v28 + 8LL * v30);
                v32 = *v31;
                if ( v9 == *v31 )
                  goto LABEL_21;
                v39 = v40;
              }
            }
          }
        }
        else
        {
          v20 = *(_QWORD **)(a1 + 32);
          v21 = (__int64)&v20[*(unsigned int *)(a1 + 40)];
          v22 = sub_27ABED0(v20, v21, v42);
          if ( (_QWORD *)v21 != v22 )
          {
            if ( (_QWORD *)v21 != v22 + 1 )
            {
              memmove(v22, v22 + 1, v21 - (_QWORD)(v22 + 1));
              v23 = *(_DWORD *)(a1 + 40);
              v2 = *(_QWORD *)(a1 + 96);
            }
            *(_DWORD *)(a1 + 40) = v23 - 1;
          }
        }
        v24 = *(unsigned int *)(a1 + 104);
        v25 = (_BYTE *)(v2 + 8 * v24);
        if ( v25 != v10 )
        {
          memmove(v4, v10, v25 - v10);
          LODWORD(v24) = *(_DWORD *)(a1 + 104);
          v2 = *(_QWORD *)(a1 + 96);
        }
        v26 = (unsigned int)(v24 - 1);
        *(_DWORD *)(a1 + 104) = v26;
        if ( v4 == (char *)(v2 + 8 * v26) )
          return;
      }
    }
    v6 = *(_QWORD **)(a2 + 32);
    v7 = &v6[*(unsigned int *)(a2 + 40)];
    v8 = v7 != sub_27ABE10(v6, (__int64)v7, v42);
    goto LABEL_4;
  }
}

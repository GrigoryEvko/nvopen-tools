// Function: sub_29A27C0
// Address: 0x29a27c0
//
void __fastcall sub_29A27C0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r12
  unsigned int v4; // esi
  __int64 v6; // rdi
  __int64 v7; // r9
  int v8; // r11d
  __int64 *v9; // r8
  unsigned int v10; // ecx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  unsigned __int64 *v14; // r8
  int v15; // eax
  __int64 v16; // rsi
  int v17; // edi
  unsigned int v18; // ecx
  __int64 *v19; // rax
  __int64 v20; // r8
  __int64 **v21; // rsi
  int v22; // eax
  int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // eax
  int v26; // edx
  __int64 v27; // rsi
  unsigned __int64 v28; // rax
  int v29; // eax
  int v30; // eax
  int v31; // eax
  __int64 v32; // rsi
  __int64 *v33; // rdi
  unsigned int v34; // r14d
  int v35; // r9d
  __int64 v36; // rcx
  int v37; // eax
  int v38; // r9d
  int v39; // r10d
  __int64 *v40; // r9

  v2 = a1[56];
  if ( v2 )
  {
    v4 = *(_DWORD *)(v2 + 120);
    v6 = v2 + 96;
    if ( v4 )
    {
      v7 = *(_QWORD *)(v2 + 104);
      v8 = 1;
      v9 = 0;
      v10 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v11 = (_QWORD *)(v7 + 16LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
      {
LABEL_4:
        v13 = v11[1];
        v14 = v11 + 1;
        if ( v13 )
          goto LABEL_5;
        goto LABEL_16;
      }
      while ( v12 != -4096 )
      {
        if ( !v9 && v12 == -8192 )
          v9 = v11;
        v10 = (v4 - 1) & (v8 + v10);
        v11 = (_QWORD *)(v7 + 16LL * v10);
        v12 = *v11;
        if ( a2 == *v11 )
          goto LABEL_4;
        ++v8;
      }
      if ( !v9 )
        v9 = v11;
      v29 = *(_DWORD *)(v2 + 112);
      ++*(_QWORD *)(v2 + 96);
      v26 = v29 + 1;
      if ( 4 * (v29 + 1) < 3 * v4 )
      {
        if ( v4 - *(_DWORD *)(v2 + 116) - v26 > v4 >> 3 )
        {
LABEL_13:
          *(_DWORD *)(v2 + 112) = v26;
          if ( *v9 != -4096 )
            --*(_DWORD *)(v2 + 116);
          *v9 = a2;
          v14 = (unsigned __int64 *)(v9 + 1);
          *v14 = 0;
LABEL_16:
          v28 = sub_D28F90((__int64 *)v2, a2, v14);
          v2 = a1[56];
          v13 = v28;
LABEL_5:
          v15 = *(_DWORD *)(v2 + 328);
          v16 = *(_QWORD *)(v2 + 312);
          if ( v15 )
          {
            v17 = v15 - 1;
            v18 = (v15 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v19 = (__int64 *)(v16 + 16LL * v18);
            v20 = *v19;
            if ( v13 == *v19 )
            {
LABEL_7:
              v21 = (__int64 **)v19[1];
LABEL_8:
              sub_2284030(v2, v21, v13, a1[58], a1[59], a1[60]);
              return;
            }
            v37 = 1;
            while ( v20 != -4096 )
            {
              v38 = v37 + 1;
              v18 = v17 & (v37 + v18);
              v19 = (__int64 *)(v16 + 16LL * v18);
              v20 = *v19;
              if ( v13 == *v19 )
                goto LABEL_7;
              v37 = v38;
            }
          }
          v21 = 0;
          goto LABEL_8;
        }
        sub_D25040(v6, v4);
        v30 = *(_DWORD *)(v2 + 120);
        if ( v30 )
        {
          v31 = v30 - 1;
          v32 = *(_QWORD *)(v2 + 104);
          v33 = 0;
          v34 = v31 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v35 = 1;
          v26 = *(_DWORD *)(v2 + 112) + 1;
          v9 = (__int64 *)(v32 + 16LL * v34);
          v36 = *v9;
          if ( a2 != *v9 )
          {
            while ( v36 != -4096 )
            {
              if ( v36 == -8192 && !v33 )
                v33 = v9;
              v34 = v31 & (v35 + v34);
              v9 = (__int64 *)(v32 + 16LL * v34);
              v36 = *v9;
              if ( a2 == *v9 )
                goto LABEL_13;
              ++v35;
            }
            if ( v33 )
              v9 = v33;
          }
          goto LABEL_13;
        }
LABEL_53:
        ++*(_DWORD *)(v2 + 112);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(v2 + 96);
    }
    sub_D25040(v6, 2 * v4);
    v22 = *(_DWORD *)(v2 + 120);
    if ( v22 )
    {
      v23 = v22 - 1;
      v24 = *(_QWORD *)(v2 + 104);
      v25 = (v22 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v26 = *(_DWORD *)(v2 + 112) + 1;
      v9 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v9;
      if ( a2 != *v9 )
      {
        v39 = 1;
        v40 = 0;
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v40 )
            v40 = v9;
          v25 = v23 & (v39 + v25);
          v9 = (__int64 *)(v24 + 16LL * v25);
          v27 = *v9;
          if ( a2 == *v9 )
            goto LABEL_13;
          ++v39;
        }
        if ( v40 )
          v9 = v40;
      }
      goto LABEL_13;
    }
    goto LABEL_53;
  }
}

// Function: sub_2090460
// Address: 0x2090460
//
void __fastcall sub_2090460(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  unsigned __int8 v7; // al
  __int64 v8; // r13
  unsigned int v9; // edx
  unsigned int v10; // r8d
  __int64 v11; // rdi
  unsigned int v12; // ebx
  unsigned int v13; // esi
  _QWORD *v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // r15
  int v18; // r10d
  int v19; // r9d
  _QWORD *v20; // r15
  __int64 v21; // rdi
  int v22; // eax
  int v23; // ecx
  unsigned int v24; // eax
  int v25; // eax
  int v26; // edx
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // r9d
  _QWORD *v31; // r8
  int v32; // eax
  int v33; // eax
  __int64 v34; // rsi
  int v35; // r8d
  _QWORD *v36; // rdi
  unsigned int v37; // ebx
  __int64 v38; // rdx
  _QWORD *v39; // rt0

  v7 = *(_BYTE *)(a2 + 16);
  if ( v7 == 17 || v7 > 0x17u )
  {
    v8 = *(_QWORD *)(a1 + 712);
    v9 = *(_DWORD *)(v8 + 232);
    if ( v9 )
    {
      v10 = v9 - 1;
      v11 = *(_QWORD *)(v8 + 216);
      v12 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
      v13 = (v9 - 1) & v12;
      v14 = (_QWORD *)(v11 + 16LL * ((v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
      v15 = *v14;
      if ( a2 == *v14 )
        return;
      v16 = *v14;
      LODWORD(v17) = (v9 - 1) & v12;
      v18 = 1;
      while ( v16 != -8 )
      {
        v17 = v10 & ((_DWORD)v17 + v18);
        v16 = *(_QWORD *)(v11 + 16 * v17);
        if ( a2 == v16 )
          return;
        ++v18;
      }
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 10 )
      {
        v19 = 1;
        v20 = 0;
        while ( v15 != -8 )
        {
          if ( v20 || v15 != -16 )
            v14 = v20;
          v13 = v10 & (v19 + v13);
          v20 = (_QWORD *)(v11 + 16LL * v13);
          v15 = *v20;
          if ( a2 == *v20 )
            goto LABEL_18;
          ++v19;
          v39 = v14;
          v14 = (_QWORD *)(v11 + 16LL * v13);
          v20 = v39;
        }
        v21 = v8 + 208;
        if ( !v20 )
          v20 = v14;
        v22 = *(_DWORD *)(v8 + 224);
        ++*(_QWORD *)(v8 + 208);
        v23 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(v8 + 228) - v23 > v9 >> 3 )
            goto LABEL_15;
          sub_1542080(v21, v9);
          v32 = *(_DWORD *)(v8 + 232);
          if ( v32 )
          {
            v33 = v32 - 1;
            v34 = *(_QWORD *)(v8 + 216);
            v35 = 1;
            v36 = 0;
            v37 = v33 & v12;
            v23 = *(_DWORD *)(v8 + 224) + 1;
            v20 = (_QWORD *)(v34 + 16LL * v37);
            v38 = *v20;
            if ( a2 != *v20 )
            {
              while ( v38 != -8 )
              {
                if ( !v36 && v38 == -16 )
                  v36 = v20;
                v37 = v33 & (v35 + v37);
                v20 = (_QWORD *)(v34 + 16LL * v37);
                v38 = *v20;
                if ( a2 == *v20 )
                  goto LABEL_15;
                ++v35;
              }
              if ( v36 )
                v20 = v36;
            }
            goto LABEL_15;
          }
          goto LABEL_53;
        }
LABEL_22:
        sub_1542080(v21, 2 * v9);
        v25 = *(_DWORD *)(v8 + 232);
        if ( v25 )
        {
          v26 = v25 - 1;
          v27 = *(_QWORD *)(v8 + 216);
          v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v23 = *(_DWORD *)(v8 + 224) + 1;
          v20 = (_QWORD *)(v27 + 16LL * v28);
          v29 = *v20;
          if ( a2 != *v20 )
          {
            v30 = 1;
            v31 = 0;
            while ( v29 != -8 )
            {
              if ( v29 == -16 && !v31 )
                v31 = v20;
              v28 = v26 & (v30 + v28);
              v20 = (_QWORD *)(v27 + 16LL * v28);
              v29 = *v20;
              if ( a2 == *v20 )
                goto LABEL_15;
              ++v30;
            }
            if ( v31 )
              v20 = v31;
          }
LABEL_15:
          *(_DWORD *)(v8 + 224) = v23;
          if ( *v20 != -8 )
            --*(_DWORD *)(v8 + 228);
          *v20 = a2;
          *((_DWORD *)v20 + 2) = 0;
LABEL_18:
          v24 = sub_1FDE000(v8, *(__int64 **)a2);
          *((_DWORD *)v20 + 2) = v24;
          v9 = v24;
          goto LABEL_19;
        }
LABEL_53:
        ++*(_DWORD *)(v8 + 224);
        BUG();
      }
      v9 = 0;
    }
    else if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 10 )
    {
      ++*(_QWORD *)(v8 + 208);
      v21 = v8 + 208;
      goto LABEL_22;
    }
LABEL_19:
    sub_208C270(a1, (__int64 *)a2, v9, a3, a4, a5);
  }
}

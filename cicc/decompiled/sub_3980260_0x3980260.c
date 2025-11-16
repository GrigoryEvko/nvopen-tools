// Function: sub_3980260
// Address: 0x3980260
//
void __fastcall sub_3980260(__int64 a1, __int64 a2, __int64 a3, __int64 a4, _QWORD *a5, _QWORD *a6)
{
  _QWORD *v7; // rdi
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // r12
  unsigned int v13; // edx
  __int64 *v14; // rbx
  __int64 *v15; // r15
  __int64 v16; // r8
  unsigned int v17; // edi
  _QWORD *v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // r12
  unsigned int v22; // edi
  _QWORD *v23; // rax
  __int64 v24; // rcx
  unsigned int v25; // esi
  __int64 v26; // r12
  int v27; // eax
  int v28; // esi
  __int64 v29; // rdi
  __int64 v30; // rax
  int v31; // ecx
  _QWORD *v32; // rdx
  __int64 v33; // r8
  int v34; // ebx
  const void *v35; // r14
  size_t v36; // r15
  int v37; // r11d
  _QWORD *v38; // rdx
  int v39; // eax
  int v40; // ecx
  int v41; // r10d
  int v42; // eax
  int v43; // eax
  int v44; // eax
  __int64 v45; // rdi
  _QWORD *v46; // r8
  __int64 v47; // r14
  int v48; // r9d
  __int64 v49; // rsi
  int v50; // eax
  int v51; // esi
  __int64 v52; // rdi
  __int64 v53; // rax
  int v54; // r10d
  int v55; // eax
  int v56; // eax
  __int64 v57; // rdi
  unsigned int v58; // r14d
  __int64 v59; // rsi
  int v60; // r10d
  _QWORD *v61; // r9
  int v62; // r10d
  __int64 i; // [rsp+18h] [rbp-68h]
  _QWORD *v64; // [rsp+20h] [rbp-60h] BYREF
  __int64 v65; // [rsp+28h] [rbp-58h]
  _QWORD v66[10]; // [rsp+30h] [rbp-50h] BYREF

  v7 = v66;
  v8 = *(_QWORD *)(a1 + 288);
  v64 = v66;
  v66[0] = v8;
  v65 = 0x400000001LL;
  v9 = 1;
  while ( v9 )
  {
    v10 = v9;
    v11 = v9 - 1;
    v12 = v7[v10 - 1];
    LODWORD(v65) = v11;
    v13 = *(_DWORD *)(v12 + 40);
    if ( v13 )
    {
      v34 = *(_DWORD *)(v12 + 40);
      v35 = *(const void **)(v12 + 32);
      v36 = 8LL * v13;
      if ( v13 > (unsigned __int64)HIDWORD(v65) - v11 )
      {
        sub_16CD150((__int64)&v64, v66, v13 + v11, 8, (int)a5, (int)a6);
        v7 = v64;
        v11 = (unsigned int)v65;
      }
      memcpy(&v7[v11], v35, v36);
      LODWORD(v65) = v34 + v65;
    }
    if ( !*(_BYTE *)(v12 + 24) )
    {
      v14 = *(__int64 **)(v12 + 80);
      v15 = &v14[2 * *(unsigned int *)(v12 + 88)];
      for ( i = a1 + 352; v15 != v14; v14 += 2 )
      {
        v25 = *(_DWORD *)(a1 + 376);
        v26 = *v14;
        if ( v25 )
        {
          v16 = *(_QWORD *)(a1 + 360);
          v17 = (v25 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v18 = (_QWORD *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( v26 == *v18 )
            goto LABEL_9;
          v41 = 1;
          v32 = 0;
          while ( v19 != -8 )
          {
            if ( v32 || v19 != -16 )
              v18 = v32;
            v17 = (v25 - 1) & (v41 + v17);
            v19 = *(_QWORD *)(v16 + 16LL * v17);
            if ( v26 == v19 )
              goto LABEL_9;
            ++v41;
            v32 = v18;
            v18 = (_QWORD *)(v16 + 16LL * v17);
          }
          if ( !v32 )
            v32 = v18;
          v42 = *(_DWORD *)(a1 + 368);
          ++*(_QWORD *)(a1 + 352);
          v31 = v42 + 1;
          if ( 4 * (v42 + 1) < 3 * v25 )
          {
            if ( v25 - *(_DWORD *)(a1 + 372) - v31 <= v25 >> 3 )
            {
              sub_39800A0(i, v25);
              v43 = *(_DWORD *)(a1 + 376);
              if ( !v43 )
              {
LABEL_95:
                ++*(_DWORD *)(a1 + 368);
                BUG();
              }
              v44 = v43 - 1;
              v45 = *(_QWORD *)(a1 + 360);
              v46 = 0;
              LODWORD(v47) = v44 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
              v48 = 1;
              v31 = *(_DWORD *)(a1 + 368) + 1;
              v32 = (_QWORD *)(v45 + 16LL * (unsigned int)v47);
              v49 = *v32;
              if ( v26 != *v32 )
              {
                while ( v49 != -8 )
                {
                  if ( v49 == -16 && !v46 )
                    v46 = v32;
                  v47 = v44 & (unsigned int)(v47 + v48);
                  v32 = (_QWORD *)(v45 + 16 * v47);
                  v49 = *v32;
                  if ( v26 == *v32 )
                    goto LABEL_16;
                  ++v48;
                }
                if ( v46 )
                  v32 = v46;
              }
            }
            goto LABEL_16;
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 352);
        }
        sub_39800A0(i, 2 * v25);
        v27 = *(_DWORD *)(a1 + 376);
        if ( !v27 )
          goto LABEL_95;
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a1 + 360);
        LODWORD(v30) = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v31 = *(_DWORD *)(a1 + 368) + 1;
        v32 = (_QWORD *)(v29 + 16LL * (unsigned int)v30);
        v33 = *v32;
        if ( v26 != *v32 )
        {
          v60 = 1;
          v61 = 0;
          while ( v33 != -8 )
          {
            if ( v33 == -16 && !v61 )
              v61 = v32;
            v30 = v28 & (unsigned int)(v30 + v60);
            v32 = (_QWORD *)(v29 + 16 * v30);
            v33 = *v32;
            if ( v26 == *v32 )
              goto LABEL_16;
            ++v60;
          }
          if ( v61 )
            v32 = v61;
        }
LABEL_16:
        *(_DWORD *)(a1 + 368) = v31;
        if ( *v32 != -8 )
          --*(_DWORD *)(a1 + 372);
        *v32 = v26;
        v32[1] = 0;
LABEL_9:
        v20 = *(_DWORD *)(a1 + 408);
        v21 = v14[1];
        if ( !v20 )
        {
          ++*(_QWORD *)(a1 + 384);
          goto LABEL_47;
        }
        LODWORD(a6) = v20 - 1;
        a5 = *(_QWORD **)(a1 + 392);
        v22 = (v20 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v23 = &a5[2 * v22];
        v24 = *v23;
        if ( v21 != *v23 )
        {
          v37 = 1;
          v38 = 0;
          while ( v24 != -8 )
          {
            if ( v38 || v24 != -16 )
              v23 = v38;
            v22 = (unsigned int)a6 & (v37 + v22);
            v24 = a5[2 * v22];
            if ( v21 == v24 )
              goto LABEL_11;
            ++v37;
            v38 = v23;
            v23 = &a5[2 * v22];
          }
          if ( !v38 )
            v38 = v23;
          v39 = *(_DWORD *)(a1 + 400);
          ++*(_QWORD *)(a1 + 384);
          v40 = v39 + 1;
          if ( 4 * (v39 + 1) < 3 * v20 )
          {
            if ( v20 - *(_DWORD *)(a1 + 404) - v40 <= v20 >> 3 )
            {
              sub_39800A0(a1 + 384, v20);
              v55 = *(_DWORD *)(a1 + 408);
              if ( !v55 )
              {
LABEL_96:
                ++*(_DWORD *)(a1 + 400);
                BUG();
              }
              v56 = v55 - 1;
              v57 = *(_QWORD *)(a1 + 392);
              a5 = 0;
              v58 = v56 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
              LODWORD(a6) = 1;
              v40 = *(_DWORD *)(a1 + 400) + 1;
              v38 = (_QWORD *)(v57 + 16LL * v58);
              v59 = *v38;
              if ( v21 != *v38 )
              {
                while ( v59 != -8 )
                {
                  if ( !a5 && v59 == -16 )
                    a5 = v38;
                  v62 = (_DWORD)a6 + 1;
                  LODWORD(a6) = v56 & (v58 + (_DWORD)a6);
                  v58 = (unsigned int)a6;
                  v38 = (_QWORD *)(v57 + 16LL * (unsigned int)a6);
                  v59 = *v38;
                  if ( v21 == *v38 )
                    goto LABEL_28;
                  LODWORD(a6) = v62;
                }
                if ( a5 )
                  v38 = a5;
              }
            }
            goto LABEL_28;
          }
LABEL_47:
          sub_39800A0(a1 + 384, 2 * v20);
          v50 = *(_DWORD *)(a1 + 408);
          if ( !v50 )
            goto LABEL_96;
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 392);
          LODWORD(v53) = (v50 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v40 = *(_DWORD *)(a1 + 400) + 1;
          v38 = (_QWORD *)(v52 + 16LL * (unsigned int)v53);
          a5 = (_QWORD *)*v38;
          if ( v21 != *v38 )
          {
            v54 = 1;
            a6 = 0;
            while ( a5 != (_QWORD *)-8LL )
            {
              if ( a5 == (_QWORD *)-16LL && !a6 )
                a6 = v38;
              v53 = v51 & (unsigned int)(v53 + v54);
              v38 = (_QWORD *)(v52 + 16 * v53);
              a5 = (_QWORD *)*v38;
              if ( v21 == *v38 )
                goto LABEL_28;
              ++v54;
            }
            if ( a6 )
              v38 = a6;
          }
LABEL_28:
          *(_DWORD *)(a1 + 400) = v40;
          if ( *v38 != -8 )
            --*(_DWORD *)(a1 + 404);
          *v38 = v21;
          v38[1] = 0;
        }
LABEL_11:
        ;
      }
    }
    v9 = v65;
    v7 = v64;
  }
  if ( v7 != v66 )
    _libc_free((unsigned __int64)v7);
}

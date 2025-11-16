// Function: sub_1EAD090
// Address: 0x1ead090
//
__int64 __fastcall sub_1EAD090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned __int64 a6)
{
  __int64 (*v7)(void); // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned int v12; // r12d
  __int64 v13; // rbx
  __int64 i; // r13
  unsigned int v15; // eax
  __int64 v17; // r12
  __int64 *v18; // rsi
  __int64 v19; // rdi
  unsigned int v20; // eax
  int v21; // ebx
  __int64 v22; // r11
  __int64 v23; // rax
  __int64 v24; // r10
  __int64 v25; // rax
  char v26; // r12
  int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // r9
  _WORD *v31; // r15
  unsigned __int16 *v32; // rdi
  _WORD *v33; // r15
  unsigned int v34; // r15d
  unsigned int j; // esi
  bool v36; // cf
  unsigned __int16 *v37; // r15
  int v38; // edi
  unsigned int v39; // ebx
  int v40; // esi
  __int64 v41; // rbx
  __int64 v42; // rsi
  unsigned __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rdx
  int v46; // esi
  __int64 v47; // [rsp+8h] [rbp-68h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+28h] [rbp-48h]
  _QWORD v51[7]; // [rsp+38h] [rbp-38h] BYREF

  v7 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v8 = 0;
  if ( v7 != sub_1D00B00 )
    v8 = v7();
  *(_QWORD *)(a1 + 232) = v8;
  v9 = *(_QWORD *)(**(_QWORD **)(a2 + 16) + 112LL);
  v10 = 0;
  if ( (__int64 (*)())v9 != sub_1D00B10 )
    v10 = ((__int64 (*)(void))v9)();
  *(_QWORD *)(a1 + 240) = v10;
  v11 = a2 + 320;
  v12 = 0;
  v47 = a2 + 320;
  *(_QWORD *)(a1 + 248) = *(_QWORD *)(a2 + 40);
  v48 = *(_QWORD *)(a2 + 328);
  if ( v48 == a2 + 320 )
    return v12;
  do
  {
    v13 = *(_QWORD *)(v48 + 32);
    for ( i = v48 + 24; i != v13; v13 = *(_QWORD *)(v13 + 8) )
    {
      while ( **(_WORD **)(v13 + 16) != 9 )
      {
        v13 = *(_QWORD *)(v13 + 8);
        if ( i == v13 )
          goto LABEL_11;
      }
      v51[0] = v13;
      sub_1EACD70(a1 + 256, v51, v9, v11, (_QWORD *)a5, (_QWORD *)a6);
    }
LABEL_11:
    v15 = *(_DWORD *)(a1 + 408);
    if ( !v15 )
      goto LABEL_12;
    do
    {
      while ( 1 )
      {
        v9 = v15;
        v17 = *(_QWORD *)(*(_QWORD *)(a1 + 400) + 8LL * v15 - 8);
        if ( (*(_BYTE *)(a1 + 264) & 1) != 0 )
        {
          a5 = a1 + 272;
          v11 = 15;
        }
        else
        {
          v11 = *(unsigned int *)(a1 + 280);
          a5 = *(_QWORD *)(a1 + 272);
          if ( !(_DWORD)v11 )
            goto LABEL_18;
          v11 = (unsigned int)(v11 - 1);
        }
        v9 = (unsigned int)v11 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v18 = (__int64 *)(a5 + 8 * v9);
        v19 = *v18;
        if ( v17 == *v18 )
        {
LABEL_17:
          *v18 = -16;
          v20 = *(_DWORD *)(a1 + 264);
          ++*(_DWORD *)(a1 + 268);
          v9 = 2 * (v20 >> 1) - 2;
          *(_DWORD *)(a1 + 264) = v9 | v20 & 1;
          v15 = *(_DWORD *)(a1 + 408);
        }
        else
        {
          v46 = 1;
          while ( v19 != -8 )
          {
            a6 = (unsigned int)(v46 + 1);
            v9 = (unsigned int)v11 & (v46 + (_DWORD)v9);
            v18 = (__int64 *)(a5 + 8LL * (unsigned int)v9);
            v19 = *v18;
            if ( v17 == *v18 )
              goto LABEL_17;
            v46 = a6;
          }
        }
LABEL_18:
        *(_DWORD *)(a1 + 408) = v15 - 1;
        v21 = *(_DWORD *)(*(_QWORD *)(v17 + 32) + 8LL);
        if ( v21 < 0 )
        {
          v41 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 248) + 24LL) + 16LL * (v21 & 0x7FFFFFFF) + 8);
          if ( v41 )
          {
            while ( (*(_BYTE *)(v41 + 3) & 0x10) != 0 || (*(_BYTE *)(v41 + 4) & 8) != 0 )
            {
              v41 = *(_QWORD *)(v41 + 32);
              if ( !v41 )
                goto LABEL_48;
            }
LABEL_55:
            v42 = *(_QWORD *)(v41 + 16);
            *(_BYTE *)(v41 + 4) |= 1u;
            v51[0] = v42;
            v43 = **(unsigned __int16 **)(v42 + 16);
            if ( (_WORD)v43 == 15
              || (_WORD)v43 == 10
              || (unsigned __int16)v43 <= 0x2Du && (v11 = 0x200000004101LL, _bittest64(&v11, v43)) )
            {
              v44 = *(_BYTE **)(v42 + 32);
              v45 = (__int64)&v44[40 * *(unsigned int *)(v42 + 40)];
              if ( v44 == (_BYTE *)v45 )
              {
LABEL_69:
                *(_QWORD *)(v42 + 16) = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 8LL) + 576LL;
                sub_1EACD70(a1 + 256, v51, v45, v11, (_QWORD *)a5, (_QWORD *)a6);
              }
              else
              {
                while ( 1 )
                {
                  if ( !*v44 && (v44[3] & 0x10) == 0 )
                  {
                    v11 = (unsigned __int8)v44[4];
                    if ( (v11 & 1) == 0 )
                    {
                      v11 &= 2u;
                      if ( !(_DWORD)v11 )
                        break;
                    }
                  }
                  v44 += 40;
                  if ( (_BYTE *)v45 == v44 )
                    goto LABEL_69;
                }
              }
            }
            while ( 1 )
            {
              v41 = *(_QWORD *)(v41 + 32);
              if ( !v41 )
                break;
              while ( (*(_BYTE *)(v41 + 3) & 0x10) == 0 )
              {
                if ( (*(_BYTE *)(v41 + 4) & 8) == 0 )
                  goto LABEL_55;
                v41 = *(_QWORD *)(v41 + 32);
                if ( !v41 )
                  goto LABEL_48;
              }
            }
          }
          goto LABEL_48;
        }
        v22 = *(_QWORD *)(v17 + 8);
        v50 = *(_QWORD *)(v17 + 24) + 24LL;
        if ( v50 == v22 )
          break;
        v49 = v17;
        while ( 1 )
        {
          v23 = *(_QWORD *)(v22 + 32);
          v9 = 5LL * *(unsigned int *)(v22 + 40);
          v24 = v23 + 40LL * *(unsigned int *)(v22 + 40);
          if ( v23 != v24 )
            break;
LABEL_37:
          v22 = *(_QWORD *)(v22 + 8);
          if ( v50 == v22 )
          {
            v17 = v49;
            goto LABEL_39;
          }
        }
        v25 = v23 + 40;
        v26 = 0;
        a5 = v25 - 40;
        if ( *(_BYTE *)(v25 - 40) )
          goto LABEL_35;
LABEL_23:
        v27 = *(_DWORD *)(v25 - 32);
        if ( v27 <= 0 )
          goto LABEL_35;
        if ( v21 != v27 )
        {
          v28 = *(_QWORD *)(a1 + 240);
          v29 = *(_QWORD *)(v28 + 8);
          v30 = *(_QWORD *)(v28 + 56);
          LODWORD(v28) = *(_DWORD *)(v29 + 24LL * (unsigned int)v21 + 16);
          v9 = v21 * (unsigned int)(v28 & 0xF);
          v31 = (_WORD *)(v30 + 2LL * ((unsigned int)v28 >> 4));
          LOWORD(v9) = *v31 + v21 * (v28 & 0xF);
          v32 = v31 + 1;
          LODWORD(v31) = *(_DWORD *)(v29 + 24LL * (unsigned int)v27 + 16);
          v11 = v27 * ((unsigned __int8)v31 & 0xFu);
          v33 = (_WORD *)(v30 + 2LL * ((unsigned int)v31 >> 4));
          LOWORD(v11) = *v33 + v11;
          a6 = (unsigned __int64)(v33 + 1);
          v34 = (unsigned __int16)v9;
          for ( j = (unsigned __int16)v11; ; j = (unsigned __int16)v11 )
          {
            v36 = v34 < j;
            if ( v34 == j )
              break;
            while ( v36 )
            {
              v37 = v32 + 1;
              v38 = *v32;
              v9 = (unsigned int)(v38 + v9);
              if ( !(_WORD)v38 )
                goto LABEL_35;
              v32 = v37;
              v34 = (unsigned __int16)v9;
              v36 = (unsigned __int16)v9 < j;
              if ( (unsigned __int16)v9 == j )
                goto LABEL_30;
            }
            v40 = *(unsigned __int16 *)a6;
            if ( !(_WORD)v40 )
            {
LABEL_35:
              while ( v24 != v25 )
              {
LABEL_34:
                v25 += 40;
                a5 = v25 - 40;
                if ( !*(_BYTE *)(v25 - 40) )
                  goto LABEL_23;
              }
              if ( v26 )
                goto LABEL_47;
              goto LABEL_37;
            }
            v11 = (unsigned int)(v40 + v11);
            a6 += 2LL;
          }
        }
LABEL_30:
        if ( (*(_BYTE *)(a5 + 3) & 0x10) == 0 )
          *(_BYTE *)(a5 + 4) |= 1u;
        if ( v24 != v25 )
        {
          v26 = 1;
          goto LABEL_34;
        }
LABEL_47:
        v17 = v49;
LABEL_48:
        sub_1E16240(v17);
        v15 = *(_DWORD *)(a1 + 408);
        if ( !v15 )
          goto LABEL_42;
      }
LABEL_39:
      v39 = *(_DWORD *)(v17 + 40) - 1;
      if ( *(_DWORD *)(v17 + 40) != 1 )
      {
        do
          sub_1E16C90(v17, v39--, v9, v11, a5, (_BYTE *)a6);
        while ( v39 );
      }
      v15 = *(_DWORD *)(a1 + 408);
    }
    while ( v15 );
LABEL_42:
    v12 = 1;
LABEL_12:
    v48 = *(_QWORD *)(v48 + 8);
  }
  while ( v47 != v48 );
  return v12;
}

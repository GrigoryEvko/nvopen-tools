// Function: sub_1D8C1E0
// Address: 0x1d8c1e0
//
__int64 __fastcall sub_1D8C1E0(__int64 a1, __int64 a2)
{
  __int64 (*v3)(void); // rdx
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  unsigned int v7; // r13d
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r12
  __int16 *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rdi
  __int64 (*v14)(); // rdx
  __int16 v15; // ax
  unsigned int v17; // eax
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rdx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  __int64 i; // rdi
  __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // r13
  _QWORD *v28; // rbx
  _QWORD *v29; // r15
  _QWORD *v30; // r12
  unsigned __int64 *v31; // rdi
  unsigned __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r8
  __int64 j; // r8
  __int64 v36; // r13
  unsigned int v37; // eax
  _BYTE *v38; // rdx
  _BYTE *v39; // r13
  _BYTE *v40; // rbx
  _BYTE *v41; // rsi
  __int64 v42; // [rsp+8h] [rbp-68h]
  int v43; // [rsp+14h] [rbp-5Ch]
  unsigned int v44; // [rsp+18h] [rbp-58h]
  unsigned __int64 v45; // [rsp+18h] [rbp-58h]
  int v46; // [rsp+20h] [rbp-50h]
  __int64 v47; // [rsp+20h] [rbp-50h]
  __int64 v48; // [rsp+28h] [rbp-48h]
  unsigned __int8 v49; // [rsp+28h] [rbp-48h]
  _BYTE *v50; // [rsp+28h] [rbp-48h]
  unsigned __int8 v51; // [rsp+28h] [rbp-48h]
  __int64 v52; // [rsp+30h] [rbp-40h]

  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 112LL);
  v4 = 0;
  if ( v3 != sub_1D00B10 )
    v4 = v3();
  *(_QWORD *)(a1 + 232) = v4;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 40LL);
  v6 = 0;
  if ( v5 != sub_1D00B00 )
    v6 = v5();
  *(_QWORD *)(a1 + 240) = v6;
  v7 = 0;
  v42 = a2 + 320;
  v52 = *(_QWORD *)(a2 + 328);
  if ( v52 != a2 + 320 )
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v52 + 32);
      if ( v8 != v52 + 24 )
        break;
LABEL_16:
      v52 = *(_QWORD *)(v52 + 8);
      if ( v42 == v52 )
        return v7;
    }
    while ( 1 )
    {
      if ( !v8 )
        BUG();
      v9 = v8;
      if ( (*(_BYTE *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 46) & 8) != 0 )
      {
        do
          v9 = *(_QWORD *)(v9 + 8);
        while ( (*(_BYTE *)(v9 + 46) & 8) != 0 );
      }
      v10 = *(_QWORD *)(v9 + 8);
      v11 = *(__int16 **)(v8 + 16);
      v12 = (*((_QWORD *)v11 + 1) >> 2) & 1LL;
      if ( ((*((_QWORD *)v11 + 1) >> 2) & 1) == 0 )
        goto LABEL_15;
      v13 = *(_QWORD *)(a1 + 240);
      v14 = *(__int64 (**)())(*(_QWORD *)v13 + 424LL);
      if ( v14 != sub_1D8C060 )
      {
        v17 = ((__int64 (__fastcall *)(__int64, __int64))v14)(v13, v8);
        if ( (_BYTE)v17 )
        {
          v7 = v17;
          goto LABEL_15;
        }
        v11 = *(__int16 **)(v8 + 16);
      }
      v15 = *v11;
      if ( v15 == 10 )
      {
        v48 = *(_QWORD *)(v8 + 24);
        v21 = *(_QWORD *)(v8 + 32);
        v43 = *(_DWORD *)(v21 + 88);
        v44 = *(_DWORD *)(v21 + 8);
        v46 = sub_38D6F10(*(_QWORD *)(a1 + 232) + 8LL, v44, *(_QWORD *)(v21 + 144));
        v7 = sub_1E17E50(v8);
        if ( (_BYTE)v7 )
        {
LABEL_42:
          *(_QWORD *)(v8 + 16) = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL) + 384LL;
          goto LABEL_15;
        }
        if ( v43 == v46 )
        {
          if ( v44 != v43 )
          {
            v7 = v12;
            *(_QWORD *)(v8 + 16) = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL) + 384LL;
            sub_1E16C90(v8, 3);
            sub_1E16C90(v8, 1);
            goto LABEL_15;
          }
        }
        else
        {
          (*(void (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 240) + 392LL))(
            *(_QWORD *)(a1 + 240),
            v48,
            v8,
            v8 + 64);
          v22 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v22 )
            BUG();
          v23 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v22 & 4) == 0 && (*(_BYTE *)(v22 + 46) & 4) != 0 )
          {
            for ( i = *(_QWORD *)v22; ; i = *(_QWORD *)v23 )
            {
              v23 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v23 + 46) & 4) == 0 )
                break;
            }
          }
          sub_1E1B830(v23, v44, 0);
        }
        v25 = v8;
        if ( (*(_BYTE *)v8 & 4) == 0 && (*(_BYTE *)(v8 + 46) & 8) != 0 )
        {
          do
            v25 = *(_QWORD *)(v25 + 8);
          while ( (*(_BYTE *)(v25 + 46) & 8) != 0 );
        }
        v26 = *(_QWORD **)(v25 + 8);
        v27 = v48 + 16;
        if ( (_QWORD *)v8 != v26 )
        {
          v47 = v10;
          v49 = v12;
          v28 = (_QWORD *)v8;
          v29 = v26;
          do
          {
            v30 = v28;
            v28 = (_QWORD *)v28[1];
            sub_1DD5BC0(v27, v30);
            v31 = (unsigned __int64 *)v30[1];
            v32 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
            *v31 = v32 | *v31 & 7;
            *(_QWORD *)(v32 + 8) = v31;
            *v30 &= 7uLL;
            v30[1] = 0;
            sub_1DD5C20(v27, v30);
          }
          while ( v28 != v29 );
          LODWORD(v12) = v49;
          v10 = v47;
        }
        goto LABEL_25;
      }
      if ( v15 == 15 )
      {
        v7 = sub_1E17E50(v8);
        if ( (_BYTE)v7 )
          goto LABEL_42;
        v18 = *(_QWORD *)(v8 + 32);
        v19 = *(_BYTE *)(v18 + 44) & 1;
        if ( *(_DWORD *)(v18 + 48) == *(_DWORD *)(v18 + 8) )
        {
          if ( v19 || *(_DWORD *)(v8 + 40) > 2u )
          {
LABEL_46:
            v7 = v12;
            *(_QWORD *)(v8 + 16) = *(_QWORD *)(*(_QWORD *)(a1 + 240) + 8LL) + 384LL;
            goto LABEL_15;
          }
        }
        else
        {
          if ( v19 )
            goto LABEL_46;
          (*(void (__fastcall **)(_QWORD, _QWORD, __int64, __int64))(**(_QWORD **)(a1 + 240) + 392LL))(
            *(_QWORD *)(a1 + 240),
            *(_QWORD *)(v8 + 24),
            v8,
            v8 + 64);
          v20 = *(unsigned int *)(v8 + 40);
          if ( (unsigned int)v20 > 2 )
          {
            v33 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
            if ( !v33 )
              BUG();
            v34 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_QWORD *)v33 & 4) == 0 && (*(_BYTE *)(v33 + 46) & 4) != 0 )
            {
              for ( j = *(_QWORD *)v33; ; j = *(_QWORD *)v34 )
              {
                v34 = j & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v34 + 46) & 4) == 0 )
                  break;
              }
            }
            v36 = *(_QWORD *)(v8 + 32);
            v45 = v34;
            v50 = (_BYTE *)(v36 + 40 * v20);
            v37 = sub_1E16360(v8);
            v38 = v50;
            v39 = (_BYTE *)(v36 + 40LL * v37);
            if ( v50 != v39 )
            {
              v51 = v12;
              v40 = v38;
              do
              {
                while ( *v39 )
                {
                  v39 += 40;
                  if ( v40 == v39 )
                    goto LABEL_60;
                }
                v41 = v39;
                v39 += 40;
                sub_1E1AFD0(v45, v41);
              }
              while ( v40 != v39 );
LABEL_60:
              LODWORD(v12) = v51;
            }
          }
        }
        sub_1E16240(v8);
LABEL_25:
        v7 = v12;
      }
LABEL_15:
      v8 = v10;
      if ( v10 == v52 + 24 )
        goto LABEL_16;
    }
  }
  return v7;
}

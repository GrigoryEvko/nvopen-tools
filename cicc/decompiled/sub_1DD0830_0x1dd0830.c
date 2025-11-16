// Function: sub_1DD0830
// Address: 0x1dd0830
//
void __fastcall sub_1DD0830(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  char *v6; // r14
  __int64 v7; // r13
  int v8; // r15d
  unsigned int v9; // ebx
  _BYTE **v10; // r11
  __int64 v11; // rax
  int v12; // r12d
  char v13; // dl
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r15
  unsigned int v18; // esi
  __int64 v19; // rax
  __int64 v20; // r12
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r15
  int v25; // esi
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // [rsp+20h] [rbp-A0h]
  _BYTE **v29; // [rsp+20h] [rbp-A0h]
  _BYTE **v30; // [rsp+20h] [rbp-A0h]
  _BYTE **v31; // [rsp+20h] [rbp-A0h]
  _BYTE *v33; // [rsp+30h] [rbp-90h] BYREF
  __int64 v34; // [rsp+38h] [rbp-88h]
  _BYTE v35[16]; // [rsp+40h] [rbp-80h] BYREF
  _BYTE *v36; // [rsp+50h] [rbp-70h] BYREF
  __int64 v37; // [rsp+58h] [rbp-68h]
  _BYTE v38[16]; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v39; // [rsp+70h] [rbp-50h] BYREF
  __int64 v40; // [rsp+78h] [rbp-48h]
  _BYTE v41[64]; // [rsp+80h] [rbp-40h] BYREF

  v6 = a1;
  v7 = a2;
  if ( **(_WORD **)(a2 + 16) != 45 && **(_WORD **)(a2 + 16) )
  {
    v8 = *(_DWORD *)(a2 + 40);
    v36 = v38;
    v37 = 0x400000000LL;
    v40 = 0x400000000LL;
    v33 = v35;
    v39 = v41;
    v34 = 0x100000000LL;
    if ( !v8 )
      goto LABEL_36;
  }
  else
  {
    v8 = 1;
    v36 = v38;
    v37 = 0x400000000LL;
    v40 = 0x400000000LL;
    v33 = v35;
    v39 = v41;
    v34 = 0x100000000LL;
  }
  v9 = 0;
  v10 = &v39;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(a2 + 32) + 40LL * v9;
      if ( *(_BYTE *)v11 == 12 )
        break;
      if ( *(_BYTE *)v11 || (v12 = *(_DWORD *)(v11 + 8)) == 0 )
      {
LABEL_18:
        if ( ++v9 == v8 )
          goto LABEL_19;
      }
      else
      {
        v13 = *(_BYTE *)(v11 + 3);
        if ( (v13 & 0x10) == 0 )
        {
          if ( v12 <= 0
            || (*(_QWORD *)(*(_QWORD *)(*((_QWORD *)a1 + 44) + 304LL) + 8LL * ((unsigned int)v12 >> 6)) & (1LL << v12)) == 0 )
          {
            *(_BYTE *)(v11 + 3) &= ~0x40u;
          }
          if ( (*(_BYTE *)(v11 + 4) & 1) == 0
            && (*(_BYTE *)(v11 + 4) & 2) == 0
            && ((*(_BYTE *)(v11 + 3) & 0x10) == 0 || (*(_DWORD *)v11 & 0xFFF00) != 0) )
          {
            v14 = (unsigned int)v37;
            if ( (unsigned int)v37 >= HIDWORD(v37) )
            {
              v31 = v10;
              sub_16CD150((__int64)&v36, v38, 0, 4, a5, a6);
              v14 = (unsigned int)v37;
              v10 = v31;
            }
            *(_DWORD *)&v36[4 * v14] = v12;
            LODWORD(v37) = v37 + 1;
          }
          goto LABEL_18;
        }
        if ( v12 <= 0
          || (a5 = *(_QWORD *)(*((_QWORD *)a1 + 44) + 304LL),
              (*(_QWORD *)(a5 + 8LL * ((unsigned int)v12 >> 6)) & (1LL << v12)) != 0) )
        {
          v26 = (unsigned int)v40;
          if ( (unsigned int)v40 >= HIDWORD(v40) )
            goto LABEL_54;
        }
        else
        {
          *(_BYTE *)(v11 + 3) = v13 & 0xBF;
          v26 = (unsigned int)v40;
          if ( (unsigned int)v40 >= HIDWORD(v40) )
          {
LABEL_54:
            v29 = v10;
            sub_16CD150((__int64)v10, v41, 0, 4, a5, a6);
            v26 = (unsigned int)v40;
            v10 = v29;
          }
        }
        ++v9;
        *(_DWORD *)&v39[4 * v26] = v12;
        LODWORD(v40) = v40 + 1;
        if ( v9 == v8 )
          goto LABEL_19;
      }
    }
    v27 = (unsigned int)v34;
    if ( (unsigned int)v34 >= HIDWORD(v34) )
    {
      v30 = v10;
      sub_16CD150((__int64)&v33, v35, 0, 4, a5, a6);
      v27 = (unsigned int)v34;
      v10 = v30;
    }
    *(_DWORD *)&v33[4 * v27] = v9++;
    LODWORD(v34) = v34 + 1;
  }
  while ( v9 != v8 );
LABEL_19:
  v28 = *(_QWORD *)(a2 + 24);
  if ( (_DWORD)v37 )
  {
    v15 = 4LL * (unsigned int)v37;
    v16 = 0;
    do
    {
      while ( 1 )
      {
        v18 = *(_DWORD *)&v36[v16];
        if ( (v18 & 0x80000000) == 0 )
          break;
        v16 += 4;
        sub_1DCCA50((__int64)a1, v18, v28, a2);
        if ( v15 == v16 )
          goto LABEL_26;
      }
      a5 = *(_QWORD *)(*((_QWORD *)a1 + 44) + 304LL);
      if ( (*(_QWORD *)(a5 + 8LL * (v18 >> 6)) & (1LL << v18)) == 0 )
        sub_1DCD8B0(a1, v18, a2, v18, a5);
      v16 += 4;
    }
    while ( v15 != v16 );
LABEL_26:
    v19 = a2;
    v6 = a1;
    v7 = v19;
  }
  v20 = 0;
  v21 = 4LL * (unsigned int)v34;
  if ( (_DWORD)v34 )
  {
    do
    {
      v22 = *(unsigned int *)&v33[v20];
      v20 += 4;
      sub_1DCFB30(v6, *(_QWORD *)(v7 + 32) + 40 * v22);
    }
    while ( v21 != v20 );
  }
  if ( (_DWORD)v40 )
  {
    v23 = 4LL * (unsigned int)v40;
    v24 = 0;
    do
    {
      while ( 1 )
      {
        v25 = *(_DWORD *)&v39[v24];
        if ( v25 >= 0 )
          break;
        v24 += 4;
        sub_1DCCC40(v6, v25, v7);
        if ( v24 == v23 )
          goto LABEL_36;
      }
      if ( (*(_QWORD *)(*(_QWORD *)(*((_QWORD *)v6 + 44) + 304LL) + 8LL * ((unsigned int)v25 >> 6)) & (1LL << v25)) == 0 )
        sub_1DCFC50(v6, v25, v7, a3, a5, a6);
      v24 += 4;
    }
    while ( v24 != v23 );
  }
LABEL_36:
  sub_1DCB440(v6, v7, a3);
  if ( v33 != v35 )
    _libc_free((unsigned __int64)v33);
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( v36 != v38 )
    _libc_free((unsigned __int64)v36);
}

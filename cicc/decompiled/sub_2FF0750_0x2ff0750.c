// Function: sub_2FF0750
// Address: 0x2ff0750
//
__int64 __fastcall sub_2FF0750(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v5; // rdx
  __int128 *v6; // rax
  __int128 *v7; // rax
  __int128 *v8; // rax
  _QWORD *v9; // r14
  _QWORD *v10; // r13
  unsigned __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rcx
  _QWORD *v17; // r8
  __int64 v18; // rax
  _QWORD *v19; // rdi
  __int64 v20; // rdx
  _QWORD *v21; // r14
  _QWORD *v22; // r13
  unsigned __int64 v23; // rsi
  _QWORD *v24; // rax
  _QWORD *v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rdx
  __int64 v28; // rax
  _QWORD *v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rdx
  __int64 (*v33)(); // rdx
  char v34; // al

  *(_QWORD *)(a1 + 16) = &unk_5027190;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 176) = a3;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_QWORD *)a1 = &unk_4A2D670;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 4;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 16777217;
  *(_QWORD *)(a1 + 256) = a2;
  *(_QWORD *)(a1 + 264) = 0;
  *(_DWORD *)(a1 + 272) = 0x10000;
  *(_WORD *)(a1 + 276) = 0;
  v5 = (_QWORD *)sub_22077B0(0x90u);
  if ( v5 )
  {
    memset(v5, 0, 0x90u);
    v5[4] = v5 + 6;
    v5[5] = 0x400000000LL;
  }
  *(_QWORD *)(a1 + 264) = v5;
  v6 = sub_BC2B00();
  sub_34CC050(v6);
  v7 = sub_BC2B00();
  sub_D05480((__int64)v7);
  v8 = sub_BC2B00();
  sub_CF6DB0((__int64)v8);
  v9 = sub_C52410();
  v10 = v9 + 1;
  v11 = sub_C959E0();
  v12 = (_QWORD *)v9[2];
  if ( v12 )
  {
    v13 = v9 + 1;
    do
    {
      while ( 1 )
      {
        v14 = v12[2];
        v15 = v12[3];
        if ( v11 <= v12[4] )
          break;
        v12 = (_QWORD *)v12[3];
        if ( !v15 )
          goto LABEL_8;
      }
      v13 = v12;
      v12 = (_QWORD *)v12[2];
    }
    while ( v14 );
LABEL_8:
    if ( v10 != v13 && v11 >= v13[4] )
      v10 = v13;
  }
  if ( v10 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v18 = v10[7];
    v17 = v10 + 6;
    if ( v18 )
    {
      v11 = (unsigned int)dword_502A048;
      v19 = v10 + 6;
      do
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(v18 + 16);
          v20 = *(_QWORD *)(v18 + 24);
          if ( *(_DWORD *)(v18 + 32) >= dword_502A048 )
            break;
          v18 = *(_QWORD *)(v18 + 24);
          if ( !v20 )
            goto LABEL_17;
        }
        v19 = (_QWORD *)v18;
        v18 = *(_QWORD *)(v18 + 16);
      }
      while ( v16 );
LABEL_17:
      if ( v17 != v19 && dword_502A048 >= *((_DWORD *)v19 + 8) && *((_DWORD *)v19 + 9) )
      {
        *(_BYTE *)(a2 + 878) = (32 * (qword_502A0C8 & 1)) | *(_BYTE *)(a2 + 878) & 0xDF;
        if ( (*(_BYTE *)(a2 + 878) & 0x20) == 0 )
          goto LABEL_21;
        goto LABEL_43;
      }
    }
  }
  v33 = *(__int64 (**)())(*(_QWORD *)a2 + 232LL);
  v34 = 0;
  if ( v33 != sub_23CE3D0 )
    v34 = ((__int64 (__fastcall *)(__int64, unsigned __int64, __int64 (*)(), __int64, _QWORD *))v33)(
            a2,
            v11,
            v33,
            v16,
            v17);
  *(_BYTE *)(a2 + 878) = *(_BYTE *)(a2 + 878) & 0xDF | (32 * (((*(_BYTE *)(a2 + 878) & 0x20) != 0) | v34 & 1));
  if ( (*(_BYTE *)(a2 + 878) & 0x20) != 0 )
LABEL_43:
    sub_2FF0740(a1, (_BYTE *)(a1 + 276), 1);
LABEL_21:
  v21 = sub_C52410();
  v22 = v21 + 1;
  v23 = sub_C959E0();
  v24 = (_QWORD *)v21[2];
  if ( v24 )
  {
    v25 = v21 + 1;
    do
    {
      while ( 1 )
      {
        v26 = v24[2];
        v27 = v24[3];
        if ( v23 <= v24[4] )
          break;
        v24 = (_QWORD *)v24[3];
        if ( !v27 )
          goto LABEL_26;
      }
      v25 = v24;
      v24 = (_QWORD *)v24[2];
    }
    while ( v26 );
LABEL_26:
    if ( v22 != v25 && v23 >= v25[4] )
      v22 = v25;
  }
  if ( v22 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v28 = v22[7];
    if ( v28 )
    {
      v29 = v22 + 6;
      do
      {
        while ( 1 )
        {
          v30 = *(_QWORD *)(v28 + 16);
          v31 = *(_QWORD *)(v28 + 24);
          if ( *(_DWORD *)(v28 + 32) >= dword_5028068 )
            break;
          v28 = *(_QWORD *)(v28 + 24);
          if ( !v31 )
            goto LABEL_35;
        }
        v29 = (_QWORD *)v28;
        v28 = *(_QWORD *)(v28 + 16);
      }
      while ( v30 );
LABEL_35:
      if ( v22 + 6 != v29 && dword_5028068 >= *((_DWORD *)v29 + 8) && *((_DWORD *)v29 + 9) )
        *(_DWORD *)(a2 + 868) = dword_50280E8;
    }
  }
  return sub_2FEF750(a1);
}

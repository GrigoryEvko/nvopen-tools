// Function: sub_2DDAF30
// Address: 0x2ddaf30
//
__int64 __fastcall sub_2DDAF30(__int64 a1, __int64 a2)
{
  _QWORD *v4; // r14
  _QWORD *v5; // r13
  unsigned __int64 v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // eax
  int v16; // edi
  __int64 v17; // r8
  char v18; // dl
  int v19; // esi
  __int16 v20; // cx
  unsigned int v21; // r13d
  __int64 v22; // rdi
  __int64 v23; // rsi
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // [rsp+0h] [rbp-F0h] BYREF
  int v29; // [rsp+8h] [rbp-E8h]
  int v30; // [rsp+Ch] [rbp-E4h]
  int v31; // [rsp+10h] [rbp-E0h]
  __int16 v32; // [rsp+14h] [rbp-DCh]
  char v33; // [rsp+16h] [rbp-DAh]
  char v34; // [rsp+18h] [rbp-D8h]
  __int64 v35; // [rsp+20h] [rbp-D0h]
  __int64 v36; // [rsp+28h] [rbp-C8h]
  __int64 v37; // [rsp+30h] [rbp-C0h]
  __int64 v38; // [rsp+38h] [rbp-B8h]
  _BYTE *v39; // [rsp+40h] [rbp-B0h]
  __int64 v40; // [rsp+48h] [rbp-A8h]
  _BYTE v41[160]; // [rsp+50h] [rbp-A0h] BYREF

  v4 = sub_C52410();
  v5 = v4 + 1;
  v6 = sub_C959E0();
  v7 = (_QWORD *)v4[2];
  if ( v7 )
  {
    v8 = v4 + 1;
    do
    {
      while ( 1 )
      {
        v9 = v7[2];
        v10 = v7[3];
        if ( v6 <= v7[4] )
          break;
        v7 = (_QWORD *)v7[3];
        if ( !v10 )
          goto LABEL_6;
      }
      v8 = v7;
      v7 = (_QWORD *)v7[2];
    }
    while ( v9 );
LABEL_6:
    if ( v5 != v8 && v6 >= v8[4] )
      v5 = v8;
  }
  if ( v5 == (_QWORD *)((char *)sub_C52410() + 8) )
    goto LABEL_25;
  v11 = v5[7];
  if ( !v11 )
    goto LABEL_25;
  v12 = v5 + 6;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(v11 + 16);
      v14 = *(_QWORD *)(v11 + 24);
      if ( *(_DWORD *)(v11 + 32) >= dword_501DA48 )
        break;
      v11 = *(_QWORD *)(v11 + 24);
      if ( !v14 )
        goto LABEL_15;
    }
    v12 = (_QWORD *)v11;
    v11 = *(_QWORD *)(v11 + 16);
  }
  while ( v13 );
LABEL_15:
  if ( v5 + 6 == v12 || dword_501DA48 < *((_DWORD *)v12 + 8) || !*((_DWORD *)v12 + 9) )
  {
LABEL_25:
    v25 = sub_BA91D0(a2, "SmallDataLimit", 0xEu);
    v15 = 0;
    if ( v25 )
    {
      v26 = *(_QWORD *)(v25 + 136);
      v27 = *(_QWORD **)(v26 + 24);
      if ( *(_DWORD *)(v26 + 32) > 0x40u )
        v27 = (_QWORD *)*v27;
      v15 = (_DWORD)v27 + 1;
      if ( !v27 )
        v15 = 0;
    }
    *(_DWORD *)(a1 + 188) = v15;
  }
  else
  {
    v15 = qword_501DAC8;
    *(_DWORD *)(a1 + 188) = qword_501DAC8;
  }
  v16 = *(_DWORD *)(a1 + 184);
  v17 = *(_QWORD *)(a1 + 176);
  v30 = v15;
  v18 = *(_BYTE *)(a1 + 198);
  v19 = *(_DWORD *)(a1 + 192);
  v34 = 0;
  v20 = *(_WORD *)(a1 + 196);
  v21 = (unsigned __int8)qword_501E0E8;
  v29 = v16;
  v28 = v17;
  v31 = v19;
  v32 = v20;
  v33 = v18;
  v35 = 0;
  v36 = 0;
  v37 = 0;
  v38 = 0;
  v39 = v41;
  v40 = 0x1000000000LL;
  if ( (_BYTE)qword_501E0E8 )
  {
    v21 = sub_2DDA340((__int64)&v28, a2);
    if ( v39 != v41 )
      _libc_free((unsigned __int64)v39);
    v22 = v36;
    v23 = 8LL * (unsigned int)v38;
  }
  else
  {
    v22 = 0;
    v23 = 0;
  }
  sub_C7D6A0(v22, v23, 8);
  return v21;
}

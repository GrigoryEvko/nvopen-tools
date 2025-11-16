// Function: sub_A1D710
// Address: 0xa1d710
//
void __fastcall sub_A1D710(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r12
  _BYTE *v5; // r14
  _BYTE *v7; // rax
  unsigned __int64 v8; // rax
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  _BYTE *v14; // rax
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  _BYTE *v19; // rax
  unsigned __int64 v20; // rax
  unsigned int v21; // eax
  __int64 v22; // rsi
  unsigned __int64 v23; // rax
  _BYTE *v24; // rax
  unsigned __int64 v25; // rax
  _BYTE *v26; // rax
  unsigned __int64 v27; // rax
  unsigned int v28; // eax
  __int64 v29; // rsi
  unsigned __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rsi
  unsigned __int64 v33; // rax
  unsigned __int8 v34; // al
  __int64 v35; // rsi
  _BYTE *v36; // r14
  unsigned __int64 v37; // rax

  v4 = a2;
  v5 = a2 - 16;
  sub_A188E0(a3, ((a2[1] & 0x7F) == 1) | 6LL);
  v7 = sub_A17150(a2 - 16);
  v8 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v7 + 1));
  sub_A188E0(a3, HIDWORD(v8));
  v9 = sub_A17150(a2 - 16);
  v10 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v9 + 2));
  sub_A188E0(a3, HIDWORD(v10));
  v11 = sub_A17150(a2 - 16);
  v12 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v11 + 3));
  sub_A188E0(a3, HIDWORD(v12));
  if ( *a2 != 16 )
    a2 = *(_BYTE **)sub_A17150(v5);
  v13 = sub_A18650((__int64)(a1 + 35), (__int64)a2);
  sub_A188E0(a3, HIDWORD(v13));
  sub_A188E0(a3, *((unsigned int *)v4 + 4));
  v14 = sub_A17150(v5);
  v15 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v14 + 4));
  sub_A188E0(a3, HIDWORD(v15));
  sub_A188E0(a3, *((unsigned int *)v4 + 5));
  if ( (*(v4 - 16) & 2) != 0 )
    v16 = *((_DWORD *)v4 - 6);
  else
    v16 = (*((_WORD *)v4 - 8) >> 6) & 0xF;
  v17 = 0;
  if ( v16 > 8 )
    v17 = *((_QWORD *)sub_A17150(v5) + 8);
  v18 = sub_A18650((__int64)(a1 + 35), v17);
  sub_A188E0(a3, HIDWORD(v18));
  sub_A188E0(a3, *((unsigned int *)v4 + 9));
  sub_A188E0(a3, *((unsigned int *)v4 + 6));
  sub_A188E0(a3, *((unsigned int *)v4 + 8));
  v19 = sub_A17150(v5);
  v20 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v19 + 5));
  sub_A188E0(a3, HIDWORD(v20));
  if ( (*(v4 - 16) & 2) != 0 )
    v21 = *((_DWORD *)v4 - 6);
  else
    v21 = (*((_WORD *)v4 - 8) >> 6) & 0xF;
  v22 = 0;
  if ( v21 > 9 )
    v22 = *((_QWORD *)sub_A17150(v5) + 9);
  v23 = sub_A18650((__int64)(a1 + 35), v22);
  sub_A188E0(a3, HIDWORD(v23));
  v24 = sub_A17150(v5);
  v25 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v24 + 6));
  sub_A188E0(a3, HIDWORD(v25));
  v26 = sub_A17150(v5);
  v27 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v26 + 7));
  sub_A188E0(a3, HIDWORD(v27));
  sub_A188E0(a3, *((int *)v4 + 7));
  if ( (*(v4 - 16) & 2) != 0 )
    v28 = *((_DWORD *)v4 - 6);
  else
    v28 = (*((_WORD *)v4 - 8) >> 6) & 0xF;
  v29 = 0;
  if ( v28 > 0xA )
    v29 = *((_QWORD *)sub_A17150(v5) + 10);
  v30 = sub_A18650((__int64)(a1 + 35), v29);
  sub_A188E0(a3, HIDWORD(v30));
  if ( (*(v4 - 16) & 2) != 0 )
    v31 = *((_DWORD *)v4 - 6);
  else
    v31 = (*((_WORD *)v4 - 8) >> 6) & 0xF;
  v32 = 0;
  if ( v31 > 0xB )
    v32 = *((_QWORD *)sub_A17150(v5) + 11);
  v33 = sub_A18650((__int64)(a1 + 35), v32);
  sub_A188E0(a3, HIDWORD(v33));
  v34 = *(v4 - 16);
  if ( (v34 & 2) == 0 )
  {
    if ( ((*((_WORD *)v4 - 8) >> 6) & 0xFu) <= 0xC )
    {
      v35 = 0;
      goto LABEL_23;
    }
    v36 = &v5[-8 * ((v34 >> 2) & 0xF)];
    goto LABEL_22;
  }
  v35 = 0;
  if ( *((_DWORD *)v4 - 6) > 0xCu )
  {
    v36 = (_BYTE *)*((_QWORD *)v4 - 4);
LABEL_22:
    v35 = *((_QWORD *)v36 + 12);
  }
LABEL_23:
  v37 = sub_A18650((__int64)(a1 + 35), v35);
  sub_A188E0(a3, HIDWORD(v37));
  sub_A1BFB0(*a1, 0x15u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

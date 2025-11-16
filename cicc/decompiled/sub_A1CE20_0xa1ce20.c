// Function: sub_A1CE20
// Address: 0xa1ce20
//
void __fastcall sub_A1CE20(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r12
  _BYTE *v6; // r14
  unsigned __int16 v7; // ax
  _BYTE *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  unsigned __int64 v12; // rax
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  unsigned int v15; // eax
  _BYTE *v16; // rax
  unsigned __int64 v17; // rax
  _BYTE *v18; // rax
  unsigned __int64 v19; // rax
  _BYTE *v20; // rax
  unsigned __int64 v21; // rax
  _BYTE *v22; // rax
  unsigned __int64 v23; // rax
  _BYTE *v24; // rax
  unsigned __int64 v25; // rax
  _BYTE *v26; // rax
  unsigned __int64 v27; // rax
  _BYTE *v28; // rax
  unsigned __int64 v29; // rax
  _BYTE *v30; // rax
  unsigned __int64 v31; // rax
  _BYTE *v32; // rax
  unsigned __int64 v33; // rax
  _BYTE *v34; // rax
  unsigned __int64 v35; // rax
  _BYTE *v36; // rax
  unsigned __int64 v37; // rax
  unsigned int v38; // esi

  v4 = a2;
  v6 = a2 - 16;
  sub_A188E0(a3, ((a2[1] & 0x7F) == 1) | 2LL);
  v7 = sub_AF18C0(a2);
  sub_A188E0(a3, v7);
  v8 = sub_A17150(a2 - 16);
  v9 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v8 + 2));
  sub_A188E0(a3, HIDWORD(v9));
  if ( *a2 != 16 )
    a2 = *(_BYTE **)sub_A17150(v6);
  v10 = sub_A18650((__int64)(a1 + 35), (__int64)a2);
  sub_A188E0(a3, HIDWORD(v10));
  sub_A188E0(a3, *((unsigned int *)v4 + 4));
  v11 = sub_A17150(v6);
  v12 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v11 + 1));
  sub_A188E0(a3, HIDWORD(v12));
  v13 = sub_A17150(v6);
  v14 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v13 + 3));
  sub_A188E0(a3, HIDWORD(v14));
  sub_A188E0(a3, *((_QWORD *)v4 + 3));
  v15 = sub_AF18D0(v4);
  sub_A188E0(a3, v15);
  sub_A188E0(a3, *((_QWORD *)v4 + 4));
  sub_A188E0(a3, *((unsigned int *)v4 + 5));
  v16 = sub_A17150(v6);
  v17 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v16 + 4));
  sub_A188E0(a3, HIDWORD(v17));
  sub_A188E0(a3, *((unsigned int *)v4 + 11));
  v18 = sub_A17150(v6);
  v19 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v18 + 5));
  sub_A188E0(a3, HIDWORD(v19));
  v20 = sub_A17150(v6);
  v21 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v20 + 6));
  sub_A188E0(a3, HIDWORD(v21));
  v22 = sub_A17150(v6);
  v23 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v22 + 7));
  sub_A188E0(a3, HIDWORD(v23));
  v24 = sub_A17150(v6);
  v25 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v24 + 8));
  sub_A188E0(a3, HIDWORD(v25));
  v26 = sub_A17150(v6);
  v27 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v26 + 9));
  sub_A188E0(a3, HIDWORD(v27));
  v28 = sub_A17150(v6);
  v29 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v28 + 10));
  sub_A188E0(a3, HIDWORD(v29));
  v30 = sub_A17150(v6);
  v31 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v30 + 11));
  sub_A188E0(a3, HIDWORD(v31));
  v32 = sub_A17150(v6);
  v33 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v32 + 12));
  sub_A188E0(a3, HIDWORD(v33));
  v34 = sub_A17150(v6);
  v35 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v34 + 13));
  sub_A188E0(a3, HIDWORD(v35));
  sub_A188E0(a3, *((unsigned int *)v4 + 10));
  v36 = sub_A17150(v6);
  v37 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v36 + 14));
  sub_A188E0(a3, HIDWORD(v37));
  v38 = -1;
  if ( v4[52] )
    v38 = *((_DWORD *)v4 + 12);
  sub_A188E0(a3, v38);
  sub_A1BFB0(*a1, 0x12u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

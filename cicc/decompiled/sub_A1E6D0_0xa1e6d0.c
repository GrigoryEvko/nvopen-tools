// Function: sub_A1E6D0
// Address: 0xa1e6d0
//
void __fastcall sub_A1E6D0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r13
  __int64 *v6; // rax
  unsigned __int64 v7; // rax
  _BYTE *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int8 v10; // al
  _BYTE *v11; // rdx
  unsigned __int64 v12; // rax
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int8 v15; // al
  _BYTE *v16; // rdx
  unsigned __int64 v17; // rax
  _BYTE *v18; // rax
  unsigned __int64 v19; // rax
  _BYTE *v20; // rax
  unsigned __int64 v21; // rax
  _BYTE *v22; // rax
  unsigned __int64 v23; // rax

  v4 = (_BYTE *)(a2 - 16);
  sub_A188E0(a3, ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) | 4LL);
  v6 = (__int64 *)sub_A17150((_BYTE *)(a2 - 16));
  v7 = sub_A18650((__int64)(a1 + 35), *v6);
  sub_A188E0(a3, HIDWORD(v7));
  v8 = sub_A17150((_BYTE *)(a2 - 16));
  v9 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v8 + 1));
  sub_A188E0(a3, HIDWORD(v9));
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(_BYTE **)(a2 - 32);
  else
    v11 = &v4[-8 * ((v10 >> 2) & 0xF)];
  v12 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v11 + 5));
  sub_A188E0(a3, HIDWORD(v12));
  v13 = sub_A17150(v4);
  v14 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v13 + 2));
  sub_A188E0(a3, HIDWORD(v14));
  sub_A188E0(a3, *(unsigned int *)(a2 + 16));
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_BYTE **)(a2 - 32);
  else
    v16 = &v4[-8 * ((v15 >> 2) & 0xF)];
  v17 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v16 + 3));
  sub_A188E0(a3, HIDWORD(v17));
  sub_A188E0(a3, *(unsigned __int8 *)(a2 + 20));
  sub_A188E0(a3, *(unsigned __int8 *)(a2 + 21));
  v18 = sub_A17150(v4);
  v19 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v18 + 6));
  sub_A188E0(a3, HIDWORD(v19));
  v20 = sub_A17150(v4);
  v21 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v20 + 7));
  sub_A188E0(a3, HIDWORD(v21));
  sub_A188E0(a3, *(unsigned int *)(a2 + 4));
  v22 = sub_A17150(v4);
  v23 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v22 + 8));
  sub_A188E0(a3, HIDWORD(v23));
  sub_A1BFB0(*a1, 0x1Bu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

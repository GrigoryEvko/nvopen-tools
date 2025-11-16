// Function: sub_A1D4A0
// Address: 0xa1d4a0
//
void __fastcall sub_A1D4A0(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r12
  unsigned __int64 v6; // rax
  _BYTE *v7; // rax
  unsigned __int64 v8; // rax
  _BYTE *v9; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  unsigned __int64 v12; // rax
  _BYTE *v13; // rax
  unsigned __int64 v14; // rax
  _BYTE *v15; // rax
  unsigned __int64 v16; // rax
  _BYTE *v17; // rax
  unsigned __int64 v18; // rax
  _BYTE *v19; // rax
  unsigned __int64 v20; // rax
  _BYTE *v21; // rax
  unsigned __int64 v22; // rax
  _BYTE *v23; // rax
  unsigned __int64 v24; // rax
  _BYTE *v25; // rax
  unsigned __int64 v26; // rax

  v4 = a2;
  sub_A188E0(a3, 1);
  sub_A188E0(a3, *(unsigned int *)(a2 + 16));
  if ( *(_BYTE *)a2 != 16 )
    a2 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  v6 = sub_A18650((__int64)(a1 + 35), a2);
  sub_A188E0(a3, HIDWORD(v6));
  v7 = sub_A17150((_BYTE *)(v4 - 16));
  v8 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v7 + 1));
  sub_A188E0(a3, HIDWORD(v8));
  sub_A188E0(a3, *(unsigned __int8 *)(v4 + 40));
  v9 = sub_A17150((_BYTE *)(v4 - 16));
  v10 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v9 + 2));
  sub_A188E0(a3, HIDWORD(v10));
  sub_A188E0(a3, *(unsigned int *)(v4 + 20));
  v11 = sub_A17150((_BYTE *)(v4 - 16));
  v12 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v11 + 3));
  sub_A188E0(a3, HIDWORD(v12));
  sub_A188E0(a3, *(unsigned int *)(v4 + 32));
  v13 = sub_A17150((_BYTE *)(v4 - 16));
  v14 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v13 + 4));
  sub_A188E0(a3, HIDWORD(v14));
  v15 = sub_A17150((_BYTE *)(v4 - 16));
  v16 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v15 + 5));
  sub_A188E0(a3, HIDWORD(v16));
  sub_A188E0(a3, 0);
  v17 = sub_A17150((_BYTE *)(v4 - 16));
  v18 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v17 + 6));
  sub_A188E0(a3, HIDWORD(v18));
  v19 = sub_A17150((_BYTE *)(v4 - 16));
  v20 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v19 + 7));
  sub_A188E0(a3, HIDWORD(v20));
  sub_A188E0(a3, *(_QWORD *)(v4 + 24));
  v21 = sub_A17150((_BYTE *)(v4 - 16));
  v22 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v21 + 8));
  sub_A188E0(a3, HIDWORD(v22));
  sub_A188E0(a3, *(unsigned __int8 *)(v4 + 41));
  sub_A188E0(a3, *(unsigned __int8 *)(v4 + 42));
  sub_A188E0(a3, *(unsigned int *)(v4 + 36));
  sub_A188E0(a3, *(unsigned __int8 *)(v4 + 43));
  v23 = sub_A17150((_BYTE *)(v4 - 16));
  v24 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v23 + 9));
  sub_A188E0(a3, HIDWORD(v24));
  v25 = sub_A17150((_BYTE *)(v4 - 16));
  v26 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v25 + 10));
  sub_A188E0(a3, HIDWORD(v26));
  sub_A1BFB0(*a1, 0x14u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

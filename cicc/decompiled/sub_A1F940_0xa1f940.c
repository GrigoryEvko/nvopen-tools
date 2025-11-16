// Function: sub_A1F940
// Address: 0xa1f940
//
void __fastcall sub_A1F940(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r12
  _BYTE *v6; // r14
  unsigned __int8 v7; // al
  _BYTE *v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  _BYTE *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned int v14; // eax
  _BYTE *v15; // rax
  unsigned __int64 v16; // rax
  _BYTE *v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int8 v19; // al
  _BYTE *v20; // rdx
  unsigned __int64 v21; // rax
  _BYTE *v22; // rax
  unsigned __int64 v23; // rax
  _BYTE *v24; // rax
  unsigned __int64 v25; // rax

  v4 = a2;
  v6 = a2 - 16;
  sub_A188E0(a3, (a2[1] & 0x7F) == 1);
  v7 = *(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = (_BYTE *)*((_QWORD *)a2 - 4);
  else
    v8 = &v6[-8 * ((v7 >> 2) & 0xF)];
  v9 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v8 + 2));
  sub_A188E0(a3, HIDWORD(v9));
  if ( *a2 != 16 )
    a2 = *(_BYTE **)sub_A17150(v6);
  v10 = sub_A18650((__int64)(a1 + 35), (__int64)a2);
  sub_A188E0(a3, HIDWORD(v10));
  sub_A188E0(a3, *((unsigned int *)v4 + 4));
  v11 = *(v4 - 16);
  if ( (v11 & 2) != 0 )
    v12 = (_BYTE *)*((_QWORD *)v4 - 4);
  else
    v12 = &v6[-8 * ((v11 >> 2) & 0xF)];
  v13 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v12 + 1));
  sub_A188E0(a3, HIDWORD(v13));
  sub_A188E0(a3, *((_QWORD *)v4 + 3));
  v14 = sub_AF18D0(v4);
  sub_A188E0(a3, v14);
  sub_A188E0(a3, *((unsigned int *)v4 + 5));
  v15 = sub_A17150(v6);
  v16 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v15 + 3));
  sub_A188E0(a3, HIDWORD(v16));
  v17 = sub_A17150(v6);
  v18 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v17 + 4));
  sub_A188E0(a3, HIDWORD(v18));
  v19 = *(v4 - 16);
  if ( (v19 & 2) != 0 )
    v20 = (_BYTE *)*((_QWORD *)v4 - 4);
  else
    v20 = &v6[-8 * ((v19 >> 2) & 0xF)];
  v21 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v20 + 5));
  sub_A188E0(a3, HIDWORD(v21));
  v22 = sub_A17150(v6);
  v23 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v22 + 6));
  sub_A188E0(a3, HIDWORD(v23));
  v24 = sub_A17150(v6);
  v25 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v24 + 7));
  sub_A188E0(a3, HIDWORD(v25));
  sub_A1BFB0(*a1, 0x30u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

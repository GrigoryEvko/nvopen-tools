// Function: sub_A1CBD0
// Address: 0xa1cbd0
//
void __fastcall sub_A1CBD0(__int64 *a1, _BYTE *a2, __int64 a3, unsigned int a4)
{
  _BYTE *v4; // r12
  _BYTE *v6; // r13
  unsigned __int16 v7; // ax
  _BYTE *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  _BYTE *v12; // rdx
  unsigned __int64 v13; // rax
  _BYTE *v14; // rax
  unsigned __int64 v15; // rax
  unsigned int v16; // eax
  _BYTE *v17; // rax
  unsigned __int64 v18; // rax
  unsigned __int8 v19; // al
  _BYTE *v20; // r13
  unsigned __int64 v21; // rsi
  __int64 v23; // [rsp+18h] [rbp-38h]

  v4 = a2;
  v6 = a2 - 16;
  sub_A188E0(a3, (a2[1] & 0x7F) == 1);
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
  v11 = *(v4 - 16);
  if ( (v11 & 2) != 0 )
    v12 = (_BYTE *)*((_QWORD *)v4 - 4);
  else
    v12 = &v6[-8 * ((v11 >> 2) & 0xF)];
  v13 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v12 + 1));
  sub_A188E0(a3, HIDWORD(v13));
  v14 = sub_A17150(v6);
  v15 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v14 + 3));
  sub_A188E0(a3, HIDWORD(v15));
  sub_A188E0(a3, *((_QWORD *)v4 + 3));
  v16 = sub_AF18D0(v4);
  sub_A188E0(a3, v16);
  sub_A188E0(a3, *((_QWORD *)v4 + 4));
  sub_A188E0(a3, *((unsigned int *)v4 + 5));
  v17 = sub_A17150(v6);
  v18 = sub_A18650((__int64)(a1 + 35), *((_QWORD *)v17 + 4));
  sub_A188E0(a3, HIDWORD(v18));
  if ( v4[48] )
    sub_A188E0(a3, (unsigned int)(*((_DWORD *)v4 + 11) + 1));
  else
    sub_A188E0(a3, 0);
  v19 = *(v4 - 16);
  if ( (v19 & 2) != 0 )
    v20 = (_BYTE *)*((_QWORD *)v4 - 4);
  else
    v20 = &v6[-8 * ((v19 >> 2) & 0xF)];
  v21 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *((_QWORD *)v20 + 5)) >> 32;
  sub_A188E0(a3, v21);
  v23 = sub_AF2E40(v4);
  if ( BYTE4(v23) )
    sub_A188E0(a3, (unsigned int)v23);
  else
    sub_A188E0(a3, 0);
  sub_A1BFB0(*a1, 0x11u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

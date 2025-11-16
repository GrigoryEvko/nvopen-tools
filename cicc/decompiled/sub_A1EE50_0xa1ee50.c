// Function: sub_A1EE50
// Address: 0xa1ee50
//
void __fastcall sub_A1EE50(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r13
  unsigned __int16 v6; // ax
  unsigned __int8 v7; // al
  __int64 *v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int8 v10; // al
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  unsigned __int64 v15; // rax
  unsigned __int8 v16; // al
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int8 v19; // al
  __int64 v20; // r13
  unsigned __int64 v21; // rax

  v5 = a2 - 16;
  sub_A188E0(a3, (*(_BYTE *)(a2 + 1) & 0x7F) == 1);
  v6 = sub_AF18C0(a2);
  sub_A188E0(a3, v6);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(__int64 **)(a2 - 32);
  else
    v8 = (__int64 *)(v5 - 8LL * ((v7 >> 2) & 0xF));
  v9 = sub_A18650((__int64)(a1 + 35), *v8);
  sub_A188E0(a3, HIDWORD(v9));
  v10 = *(_BYTE *)(a2 - 16);
  if ( (v10 & 2) != 0 )
    v11 = *(_QWORD *)(a2 - 32);
  else
    v11 = v5 - 8LL * ((v10 >> 2) & 0xF);
  v12 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v11 + 8));
  sub_A188E0(a3, HIDWORD(v12));
  sub_A188E0(a3, *(unsigned int *)(a2 + 4));
  v13 = *(_BYTE *)(a2 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(a2 - 32);
  else
    v14 = v5 - 8LL * ((v13 >> 2) & 0xF);
  v15 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v14 + 16));
  sub_A188E0(a3, HIDWORD(v15));
  v16 = *(_BYTE *)(a2 - 16);
  if ( (v16 & 2) != 0 )
    v17 = *(_QWORD *)(a2 - 32);
  else
    v17 = v5 - 8LL * ((v16 >> 2) & 0xF);
  v18 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v17 + 24));
  sub_A188E0(a3, HIDWORD(v18));
  v19 = *(_BYTE *)(a2 - 16);
  if ( (v19 & 2) != 0 )
    v20 = *(_QWORD *)(a2 - 32);
  else
    v20 = v5 - 8LL * ((v19 >> 2) & 0xF);
  v21 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v20 + 32));
  sub_A188E0(a3, HIDWORD(v21));
  sub_A1BFB0(*a1, 0x1Fu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

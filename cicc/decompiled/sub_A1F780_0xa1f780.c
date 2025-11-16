// Function: sub_A1F780
// Address: 0xa1f780
//
void __fastcall sub_A1F780(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r14
  unsigned __int8 v6; // al
  __int64 *v7; // rdx
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  unsigned __int8 v11; // al
  __int64 v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int8 v17; // al
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // [rsp+0h] [rbp-40h]

  v5 = a2 - 16;
  sub_A188E0(a3, (*(_BYTE *)(a2 + 1) & 0x7F) == 1);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(__int64 **)(a2 - 32);
  else
    v7 = (__int64 *)(v5 - 8LL * ((v6 >> 2) & 0xF));
  v8 = sub_A18650((__int64)(a1 + 35), *v7);
  v9 = *(unsigned int *)(a3 + 8);
  v10 = HIDWORD(v8);
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v20 = v10;
    sub_C8D5F0(a3, a3 + 16, v9 + 1, 8);
    v9 = *(unsigned int *)(a3 + 8);
    v10 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v9) = v10;
  ++*(_DWORD *)(a3 + 8);
  v11 = *(_BYTE *)(a2 - 16);
  if ( (v11 & 2) != 0 )
    v12 = *(_QWORD *)(a2 - 32);
  else
    v12 = v5 - 8LL * ((v11 >> 2) & 0xF);
  v13 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v12 + 8));
  sub_A188E0(a3, HIDWORD(v13));
  v14 = *(_BYTE *)(a2 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(a2 - 32);
  else
    v15 = v5 - 8LL * ((v14 >> 2) & 0xF);
  v16 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v15 + 16));
  sub_A188E0(a3, HIDWORD(v16));
  v17 = *(_BYTE *)(a2 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(_QWORD *)(a2 - 32);
  else
    v18 = v5 - 8LL * ((v17 >> 2) & 0xF);
  v19 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v18 + 24));
  sub_A188E0(a3, HIDWORD(v19));
  sub_A1BFB0(*a1, 0x2Du, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

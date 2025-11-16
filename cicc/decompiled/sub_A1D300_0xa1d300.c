// Function: sub_A1D300
// Address: 0xa1d300
//
void __fastcall sub_A1D300(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r15
  unsigned __int8 v7; // al
  __int64 *v8; // rdx
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // al
  __int64 v13; // r15
  unsigned __int64 v14; // rax
  __int64 v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // [rsp+0h] [rbp-40h]

  v5 = a2 - 16;
  sub_A188E0(a3, (*(_BYTE *)(a2 + 1) & 0x7F) == 1);
  v7 = *(_BYTE *)(a2 - 16);
  if ( (v7 & 2) != 0 )
    v8 = *(__int64 **)(a2 - 32);
  else
    v8 = (__int64 *)(v5 - 8LL * ((v7 >> 2) & 0xF));
  v9 = sub_A18650((__int64)(a1 + 35), *v8);
  v10 = *(unsigned int *)(a3 + 8);
  v11 = HIDWORD(v9);
  if ( v10 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v19 = v11;
    sub_C8D5F0(a3, a3 + 16, v10 + 1, 8);
    v10 = *(unsigned int *)(a3 + 8);
    v11 = v19;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v10) = v11;
  ++*(_DWORD *)(a3 + 8);
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v5 - 8LL * ((v12 >> 2) & 0xF);
  v14 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v13 + 8));
  sub_A188E0(a3, HIDWORD(v14));
  if ( *(_BYTE *)(a2 + 32) )
  {
    sub_A188E0(a3, *(unsigned int *)(a2 + 16));
    v15 = *(_QWORD *)(a2 + 24);
  }
  else
  {
    sub_A188E0(a3, 0);
    v15 = 0;
  }
  v16 = sub_A18650((__int64)(a1 + 35), v15);
  sub_A188E0(a3, HIDWORD(v16));
  v17 = *(_QWORD *)(a2 + 40);
  if ( v17 )
  {
    v18 = sub_A18650((__int64)(a1 + 35), v17);
    sub_A188E0(a3, HIDWORD(v18));
  }
  sub_A1BFB0(*a1, 0x10u, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

// Function: sub_A1E500
// Address: 0xa1e500
//
void __fastcall sub_A1E500(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // rax
  _BOOL8 v6; // r14
  __int64 v7; // r14
  unsigned __int16 v8; // ax
  unsigned __int8 v9; // al
  __int64 *v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rax
  unsigned __int8 v17; // al
  __int64 v18; // r14
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // [rsp+0h] [rbp-40h]

  v5 = *(unsigned int *)(a3 + 8);
  v6 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v5 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v5 + 1, 8);
    v5 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v5) = v6;
  v7 = a2 - 16;
  ++*(_DWORD *)(a3 + 8);
  v8 = sub_AF18C0(a2);
  sub_A188E0(a3, v8);
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(__int64 **)(a2 - 32);
  else
    v10 = (__int64 *)(v7 - 8LL * ((v9 >> 2) & 0xF));
  v11 = sub_A18650((__int64)(a1 + 35), *v10);
  sub_A188E0(a3, HIDWORD(v11));
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v7 - 8LL * ((v12 >> 2) & 0xF);
  v14 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v13 + 8));
  sub_A188E0(a3, HIDWORD(v14));
  v15 = *(unsigned int *)(a3 + 8);
  v16 = (unsigned __int64)*(char *)(a2 + 1) >> 63;
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v20 = (unsigned __int64)*(char *)(a2 + 1) >> 63;
    sub_C8D5F0(a3, a3 + 16, v15 + 1, 8);
    v15 = *(unsigned int *)(a3 + 8);
    v16 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v15) = v16;
  ++*(_DWORD *)(a3 + 8);
  v17 = *(_BYTE *)(a2 - 16);
  if ( (v17 & 2) != 0 )
    v18 = *(_QWORD *)(a2 - 32);
  else
    v18 = v7 - 8LL * ((v17 >> 2) & 0xF);
  v19 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v18 + 16));
  sub_A188E0(a3, HIDWORD(v19));
  sub_A1BFB0(*a1, 0x1Au, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

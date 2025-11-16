// Function: sub_A1C690
// Address: 0xa1c690
//
void __fastcall sub_A1C690(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v5; // r14
  unsigned __int8 v6; // al
  __int64 *v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  unsigned __int8 v12; // al
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  unsigned __int8 v15; // al
  __int64 v16; // r14
  unsigned __int64 v17; // r12
  __int64 v18; // rax

  v5 = a2 - 16;
  sub_A188E0(a3, ((*(_BYTE *)(a2 + 1) & 0x7F) == 1) | 4LL);
  v6 = *(_BYTE *)(a2 - 16);
  if ( (v6 & 2) != 0 )
    v7 = *(__int64 **)(a2 - 32);
  else
    v7 = (__int64 *)(v5 - 8LL * ((v6 >> 2) & 0xF));
  v8 = sub_A18650((__int64)(a1 + 35), *v7);
  sub_A188E0(a3, HIDWORD(v8));
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(_QWORD *)(a2 - 32);
  else
    v10 = v5 - 8LL * ((v9 >> 2) & 0xF);
  v11 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v10 + 8));
  sub_A188E0(a3, HIDWORD(v11));
  v12 = *(_BYTE *)(a2 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD *)(a2 - 32);
  else
    v13 = v5 - 8LL * ((v12 >> 2) & 0xF);
  v14 = sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v13 + 16));
  sub_A188E0(a3, HIDWORD(v14));
  v15 = *(_BYTE *)(a2 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(a2 - 32);
  else
    v16 = v5 - 8LL * ((v15 >> 2) & 0xF);
  v17 = (unsigned __int64)sub_A18650((__int64)(a1 + 35), *(_QWORD *)(v16 + 24)) >> 32;
  v18 = *(unsigned int *)(a3 + 8);
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v18 + 1, 8);
    v18 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v17;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 0xDu, a3, a4);
  *(_DWORD *)(a3 + 8) = 0;
}

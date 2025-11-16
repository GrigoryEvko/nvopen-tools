// Function: sub_A239F0
// Address: 0xa239f0
//
void __fastcall sub_A239F0(__int64 *a1, __int64 a2, __int64 a3, unsigned int *a4)
{
  __int64 v7; // rax
  _BOOL8 v8; // r15
  unsigned __int8 v9; // al
  __int64 *v10; // rdx
  __int64 v11; // rax
  _BYTE *v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  int v15; // eax
  __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // r12
  __int64 v20; // [rsp+8h] [rbp-38h]

  if ( !*a4 )
    *a4 = sub_A237E0(a1);
  v7 = *(unsigned int *)(a3 + 8);
  v8 = (*(_BYTE *)(a2 + 1) & 0x7F) == 1;
  if ( v7 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v7 + 1, 8);
    v7 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v7) = v8;
  ++*(_DWORD *)(a3 + 8);
  sub_A188E0(a3, *(unsigned int *)(a2 + 4));
  sub_A188E0(a3, *(unsigned __int16 *)(a2 + 2));
  v9 = *(_BYTE *)(a2 - 16);
  if ( (v9 & 2) != 0 )
    v10 = *(__int64 **)(a2 - 32);
  else
    v10 = (__int64 *)(a2 - 16 - 8LL * ((v9 >> 2) & 0xF));
  v11 = sub_A18650((__int64)(a1 + 35), *v10);
  v12 = (_BYTE *)(a2 - 16);
  v13 = (unsigned int)(HIDWORD(v11) - 1);
  v14 = *(unsigned int *)(a3 + 8);
  if ( v14 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    v20 = v13;
    sub_C8D5F0(a3, a3 + 16, v14 + 1, 8);
    v14 = *(unsigned int *)(a3 + 8);
    v12 = (_BYTE *)(a2 - 16);
    v13 = v20;
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v14) = v13;
  ++*(_DWORD *)(a3 + 8);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v15 = *(_DWORD *)(a2 - 24);
  else
    v15 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v16 = 0;
  if ( v15 == 2 )
    v16 = *((_QWORD *)sub_A17150(v12) + 1);
  v17 = sub_A18650((__int64)(a1 + 35), v16);
  sub_A188E0(a3, HIDWORD(v17));
  v18 = *(unsigned int *)(a3 + 8);
  v19 = (unsigned __int64)*(char *)(a2 + 1) >> 63;
  if ( v18 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
  {
    sub_C8D5F0(a3, a3 + 16, v18 + 1, 8);
    v18 = *(unsigned int *)(a3 + 8);
  }
  *(_QWORD *)(*(_QWORD *)a3 + 8 * v18) = v19;
  ++*(_DWORD *)(a3 + 8);
  sub_A1BFB0(*a1, 7u, a3, *a4);
  *(_DWORD *)(a3 + 8) = 0;
}

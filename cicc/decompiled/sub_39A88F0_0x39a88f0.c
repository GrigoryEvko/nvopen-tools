// Function: sub_39A88F0
// Address: 0x39a88f0
//
void __fastcall sub_39A88F0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // rax
  __int16 v8; // r14
  size_t v9; // rdx
  size_t v10; // r15
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned __int16 v13; // r15
  __int64 v14; // rax
  __int64 v15; // r8
  size_t v16; // [rsp+8h] [rbp-58h]
  void *v17; // [rsp+10h] [rbp-50h]
  __int64 v18; // [rsp+18h] [rbp-48h]
  _DWORD v19[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v5 = *(unsigned int *)(a3 + 8);
  v6 = *(_QWORD *)(a3 + 8 * (2 - v5));
  if ( v6 )
  {
    v7 = sub_161E970(v6);
    v8 = *(_WORD *)(a2 + 28);
    v17 = (void *)v7;
    v10 = v9;
    v16 = v9;
    v18 = *(_QWORD *)(a3 + 32) >> 3;
    v11 = 3LL - *(unsigned int *)(a3 + 8);
    v12 = *(_QWORD *)(a3 + 8 * v11);
    if ( !v12 )
      goto LABEL_4;
  }
  else
  {
    v17 = 0;
    v16 = 0;
    v8 = *(_WORD *)(a2 + 28);
    v18 = *(_QWORD *)(a3 + 32) >> 3;
    LODWORD(v11) = 3 - v5;
    v12 = *(_QWORD *)(a3 + 8 * (3 - v5));
    if ( !v12 )
      goto LABEL_6;
    v10 = 0;
  }
  v11 = sub_39A6760(a1, a2, v12, 73);
LABEL_4:
  if ( v10 )
    v11 = sub_39A3F30(a1, a2, 3, v17, v16);
LABEL_6:
  v13 = v8 - 15;
  LOBYTE(v11) = v18 != 0;
  if ( (unsigned __int16)(v8 - 15) <= 0x33u )
    LODWORD(v11) = ~(unsigned int)(0x8000000010003uLL >> v13) & v11;
  if ( (_BYTE)v11 )
  {
    BYTE2(v19[0]) = 0;
    sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 11, (__int64)v19, v18);
  }
  else if ( v8 == 31 )
  {
    v14 = sub_39A64F0(a1, *(_QWORD *)(a3 + 8 * (4LL - *(unsigned int *)(a3 + 8))));
    sub_39A3B20((__int64)a1, a2, 29, v14);
    if ( (*(_BYTE *)(a3 + 28) & 4) != 0 )
      goto LABEL_11;
LABEL_17:
    sub_39A3790((__int64)a1, a2, a3);
    goto LABEL_11;
  }
  if ( (*(_BYTE *)(a3 + 28) & 4) == 0 )
    goto LABEL_17;
LABEL_11:
  if ( *(_BYTE *)(a3 + 56) && (v13 <= 1u || v8 == 66) )
  {
    v15 = *(unsigned int *)(a3 + 52);
    v19[0] = 65542;
    sub_39A3560((__int64)a1, (__int64 *)(a2 + 8), 51, (__int64)v19, v15);
  }
}

// Function: sub_1A7F010
// Address: 0x1a7f010
//
unsigned __int64 __fastcall sub_1A7F010(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  unsigned int v7; // eax
  unsigned __int64 v8; // r15
  unsigned __int64 *v9; // r13
  int i; // r15d
  __int64 v11; // rax
  _QWORD *v12; // rdi
  unsigned __int64 v13; // rax
  int v14; // r8d
  int v15; // r9d
  __int64 v16; // rdx
  unsigned __int64 *v17; // rdx
  unsigned int v18; // [rsp+Ch] [rbp-64h]
  unsigned __int64 v19; // [rsp+10h] [rbp-60h]
  unsigned __int64 v20; // [rsp+18h] [rbp-58h]
  unsigned __int64 v21; // [rsp+20h] [rbp-50h]
  unsigned __int64 v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  __int64 v24[7]; // [rsp+38h] [rbp-38h] BYREF

  result = sub_157EBA0(a2);
  if ( *(_BYTE *)(result + 16) != 26 )
    return result;
  if ( (*(_DWORD *)(result + 20) & 0xFFFFFFF) != 3 )
    return result;
  result = *(_QWORD *)(result - 72);
  v21 = result;
  if ( *(_BYTE *)(result + 16) != 75 )
    return result;
  result = *(_QWORD *)(result - 24);
  if ( *(_BYTE *)(result + 16) > 0x10u )
    return result;
  v7 = *(unsigned __int16 *)(v21 + 18);
  BYTE1(v7) &= ~0x80u;
  v18 = v7;
  result = v7 - 32;
  if ( (unsigned int)result > 1 )
    return result;
  v23 = a1;
  v19 = *(_QWORD *)(v21 - 48);
  v8 = (a1 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  result = sub_1389B50(&v23);
  v9 = (unsigned __int64 *)v8;
  v22 = result;
  if ( result == v8 )
    return result;
  for ( i = 0; ; ++i )
  {
    result = *v9;
    if ( *(_BYTE *)(*v9 + 16) <= 0x10u )
      goto LABEL_13;
    v20 = v23 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = (_QWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 56);
    if ( (v23 & 4) == 0 )
      break;
    result = sub_1560290(v12, i, 32);
    if ( (_BYTE)result )
      goto LABEL_13;
    v11 = *(_QWORD *)(v20 - 24);
    if ( *(_BYTE *)(v11 + 16) )
      goto LABEL_12;
LABEL_11:
    v24[0] = *(_QWORD *)(v11 + 112);
    result = sub_1560290(v24, i, 32);
    if ( !(_BYTE)result )
      goto LABEL_12;
LABEL_13:
    v9 += 3;
    if ( (unsigned __int64 *)v22 == v9 )
      return result;
  }
  result = sub_1560290(v12, i, 32);
  if ( (_BYTE)result )
    goto LABEL_13;
  v11 = *(_QWORD *)(v20 - 72);
  if ( !*(_BYTE *)(v11 + 16) )
    goto LABEL_11;
LABEL_12:
  result = v19;
  if ( v19 != *v9 )
    goto LABEL_13;
  v13 = sub_157EBA0(a2);
  if ( a3 != sub_15F4DF0(v13, 0) )
    v18 = sub_15FF0F0(*(_WORD *)(v21 + 18) & 0x7FFF);
  v16 = *(unsigned int *)(a4 + 8);
  if ( (unsigned int)v16 >= *(_DWORD *)(a4 + 12) )
  {
    sub_16CD150(a4, (const void *)(a4 + 16), 0, 16, v14, v15);
    v16 = *(unsigned int *)(a4 + 8);
  }
  v17 = (unsigned __int64 *)(*(_QWORD *)a4 + 16 * v16);
  v17[1] = v18;
  *v17 = v21;
  ++*(_DWORD *)(a4 + 8);
  return v21;
}

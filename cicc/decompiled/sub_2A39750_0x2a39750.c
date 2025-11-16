// Function: sub_2A39750
// Address: 0x2a39750
//
_BYTE *__fastcall sub_2A39750(__int64 a1, __int64 a2)
{
  __int64 v3; // r15
  _BYTE *result; // rax
  char v5; // bl
  _QWORD *v6; // rsi
  __int64 v7; // rdx
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+18h] [rbp-48h]
  unsigned int v15[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(a2 + 80) != *(_QWORD *)(v3 + 24) )
    return sub_2A37680((__int64 **)a1, a2);
  v5 = sub_981210(**(_QWORD **)(a1 + 40), *(_QWORD *)(a2 - 32), v15);
  if ( v5 )
  {
    v6 = *(_QWORD **)(a1 + 40);
    v5 = 0;
    if ( (v6[((unsigned __int64)v15[0] >> 6) + 1] & (1LL << SLOBYTE(v15[0]))) == 0 )
      v5 = (((int)*(unsigned __int8 *)(*v6 + (v15[0] >> 2)) >> (2 * (v15[0] & 3))) & 3) != 0;
  }
  v13 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 24LL))(a1, 3);
  v12 = v7;
  v14 = *(_QWORD *)(a1 + 16);
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 32LL))(a1);
  if ( v8 == 14 )
  {
    v11 = sub_22077B0(0x1B0u);
    v10 = v11;
    if ( v11 )
      sub_B176B0(v11, v14, v13, v12, a2);
  }
  else
  {
    if ( v8 != 15 )
      BUG();
    v9 = sub_22077B0(0x1B0u);
    v10 = v9;
    if ( v9 )
      sub_B178C0(v9, v14, v13, v12, a2);
  }
  sub_2A395B0(a1, (unsigned __int8 *)v3, v5, v10);
  sub_2A38C60(a1, a2, v15[0], v10);
  result = sub_1049740(*(__int64 **)(a1 + 8), v10);
  if ( v10 )
    return (_BYTE *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL))(v10);
  return result;
}

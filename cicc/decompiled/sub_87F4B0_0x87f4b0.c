// Function: sub_87F4B0
// Address: 0x87f4b0
//
_QWORD *__fastcall sub_87F4B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // r12
  __int64 v10; // rdx

  if ( (unsigned __int8)sub_877F80(a1) == 3 )
    v5 = sub_877120(a3);
  else
    v5 = *(_QWORD *)a1;
  v9 = sub_87EBB0(((*(_BYTE *)(a1 + 81) & 0x10) == 0) + 10, v5, a2);
  *((_DWORD *)v9 + 10) = *(_DWORD *)(a1 + 40);
  v10 = *(_QWORD *)(a1 + 64);
  if ( (*(_BYTE *)(a1 + 81) & 0x10) != 0 )
  {
    sub_877E20((__int64)v9, 0, v10, v6, v7, v8);
    return v9;
  }
  else
  {
    if ( v10 )
      sub_877E90((__int64)v9, 0, v10);
    return v9;
  }
}

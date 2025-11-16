// Function: sub_1789760
// Address: 0x1789760
//
__int64 __fastcall sub_1789760(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int8 *v8; // rsi
  unsigned int v9; // ebx
  __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  _QWORD v15[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v5 = *(_QWORD *)(**(_QWORD **)(a3 - 8) + 48LL);
    v15[0] = v5;
    if ( v5 )
      goto LABEL_3;
  }
  else
  {
    v5 = *(_QWORD *)(*(_QWORD *)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF)) + 48LL);
    v15[0] = v5;
    if ( v5 )
    {
LABEL_3:
      v6 = a2 + 48;
      sub_1623A60((__int64)v15, v5, 2);
      v7 = *(_QWORD *)(a2 + 48);
      if ( !v7 )
        goto LABEL_5;
      goto LABEL_4;
    }
  }
  v7 = *(_QWORD *)(a2 + 48);
  v6 = a2 + 48;
  if ( !v7 )
    goto LABEL_7;
LABEL_4:
  sub_161E7C0(v6, v7);
LABEL_5:
  v8 = (unsigned __int8 *)v15[0];
  *(_QWORD *)(a2 + 48) = v15[0];
  if ( v8 )
    sub_1623210((__int64)v15, v8, v6);
LABEL_7:
  v9 = 1;
  result = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
  if ( (_DWORD)result != 1 )
  {
    do
    {
      if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
        v11 = *(_QWORD *)(a3 - 8);
      else
        v11 = a3 - 24 * result;
      v12 = v9++;
      v13 = sub_15C70A0(*(_QWORD *)(v11 + 24 * v12) + 48LL);
      v14 = sub_15C70A0(v6);
      sub_15AC0B0(a2, v14, v13);
      result = *(_DWORD *)(a3 + 20) & 0xFFFFFFF;
    }
    while ( v9 != (_DWORD)result );
  }
  return result;
}

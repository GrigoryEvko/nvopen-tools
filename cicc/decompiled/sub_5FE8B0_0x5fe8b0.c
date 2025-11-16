// Function: sub_5FE8B0
// Address: 0x5fe8b0
//
_BYTE *__fastcall sub_5FE8B0(__int64 *a1, unsigned int *a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // r13
  _BYTE *result; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  _BYTE v9[112]; // [rsp+10h] [rbp-2F0h] BYREF
  _QWORD v10[80]; // [rsp+80h] [rbp-280h] BYREF

  v2 = *a1;
  sub_5E4C60((__int64)v10, (_QWORD *)(*a1 + 64));
  v3 = sub_73C570(v2, a2[1], -1);
  v4 = sub_72D600(v3);
  v5 = sub_724EF0(v4);
  sub_87E3B0(v9);
  sub_5FE480(a1, (__int64)v10, (__int64)v9, v5);
  if ( a2[5] && dword_4D04464 )
  {
    v7 = v10[0];
    v8 = *(_QWORD *)(v10[0] + 88LL);
    *(_BYTE *)(v10[0] + 81LL) |= 2u;
    *(_BYTE *)(v8 + 206) |= 0x10u;
    *(_BYTE *)(*(_QWORD *)(v7 + 88) + 193LL) |= 0x20u;
  }
  result = &dword_4D0488C;
  if ( dword_4D0488C )
  {
    result = (_BYTE *)a2[13];
    if ( !(_DWORD)result )
    {
      result = *(_BYTE **)(*(_QWORD *)v2 + 96LL);
      if ( (result[183] & 4) == 0 )
      {
        result = *(_BYTE **)(v10[0] + 88LL);
        result[193] |= 2u;
      }
    }
  }
  return result;
}

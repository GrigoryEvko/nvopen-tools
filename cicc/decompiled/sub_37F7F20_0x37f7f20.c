// Function: sub_37F7F20
// Address: 0x37f7f20
//
__int64 __fastcall sub_37F7F20(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 v4; // r12
  char v5; // r15
  __int16 v6; // ax
  __int64 v8; // [rsp+8h] [rbp-68h]
  _QWORD v9[2]; // [rsp+10h] [rbp-60h] BYREF
  char v10; // [rsp+20h] [rbp-50h]
  __int64 v11; // [rsp+28h] [rbp-48h]

  v8 = *(_QWORD *)a2;
  if ( !*(_BYTE *)(a2 + 8) )
    return sub_37F51A0(a1, *(_QWORD *)a2);
  sub_37F67F0(a1, v8);
  sub_37F4460((__int64)v9, *(_QWORD *)(v8 + 56), v8 + 48, 1);
  v2 = v9[0];
  v3 = v11;
  v4 = v9[1];
  v5 = v10 ^ 1;
  while ( v3 != v2 )
  {
    sub_37F7730(a1, v2);
    do
    {
      v2 = *(_QWORD *)(v2 + 8);
      if ( v2 == v4 )
        break;
      v6 = *(_WORD *)(v2 + 68);
    }
    while ( (unsigned __int16)(v6 - 14) <= 4u || v6 == 24 && !v5 );
  }
  return sub_37F4FD0(a1, v8);
}

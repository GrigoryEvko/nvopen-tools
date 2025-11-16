// Function: sub_24C0320
// Address: 0x24c0320
//
_QWORD *__fastcall sub_24C0320(__int64 *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // r12
  char v6; // al
  __int64 v8; // [rsp+8h] [rbp-28h]

  BYTE4(v8) = 0;
  v4 = sub_BD2C40(88, unk_3F0FAE8);
  v5 = v4;
  if ( v4 )
    sub_B30000((__int64)v4, *a1, a3, 0, 9, 0, a2, 0, 0, v8, 0);
  v6 = v5[4] & 0xCF | 0x10;
  *((_BYTE *)v5 + 32) = v6;
  if ( (v6 & 0xF) != 9 )
    *((_BYTE *)v5 + 33) |= 0x40u;
  return v5;
}

// Function: sub_2A2E0D0
// Address: 0x2a2e0d0
//
_QWORD *__fastcall sub_2A2E0D0(__int64 a1)
{
  _QWORD *v1; // r12
  char v2; // al
  __int64 v4; // [rsp+8h] [rbp-48h]
  const char *v5; // [rsp+10h] [rbp-40h] BYREF
  char v6; // [rsp+30h] [rbp-20h]
  char v7; // [rsp+31h] [rbp-1Fh]

  v5 = "__dso_handle";
  v7 = 1;
  v6 = 3;
  BYTE4(v4) = 0;
  v1 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v1 )
    sub_B30000((__int64)v1, *(_QWORD *)a1, **(_QWORD ***)(a1 + 8), 1, 9, 0, (__int64)&v5, 0, 0, v4, 0);
  v2 = v1[4] & 0xCF | 0x10;
  *((_BYTE *)v1 + 32) = v2;
  if ( (v2 & 0xF) != 9 )
    *((_BYTE *)v1 + 33) |= 0x40u;
  return v1;
}

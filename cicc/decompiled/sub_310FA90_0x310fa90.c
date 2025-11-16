// Function: sub_310FA90
// Address: 0x310fa90
//
_QWORD *__fastcall sub_310FA90(__int64 *a1)
{
  __int64 v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // rax
  _QWORD *v4; // r12
  __int64 v6; // [rsp+8h] [rbp-48h]
  _QWORD v7[4]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v8; // [rsp+30h] [rbp-20h]

  v1 = a1[1];
  v8 = 261;
  BYTE4(v6) = 0;
  v2 = *(_QWORD *)(v1 + 8);
  v7[1] = *(_QWORD *)(v1 + 16);
  v7[0] = v2;
  v3 = sub_BD2C40(88, unk_3F0FAE8);
  v4 = v3;
  if ( v3 )
    sub_B30000((__int64)v3, *a1, *(_QWORD **)(a1[1] + 40), 0, 0, 0, (__int64)v7, 0, 0, v6, 0);
  *((_BYTE *)v4 + 33) |= 0x40u;
  return v4;
}

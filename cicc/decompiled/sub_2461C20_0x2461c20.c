// Function: sub_2461C20
// Address: 0x2461c20
//
_QWORD *__fastcall sub_2461C20(__int64 a1)
{
  __int64 *v1; // rax
  __int64 v2; // rdx
  _QWORD *v3; // r12
  __int64 v5; // [rsp+8h] [rbp-48h]
  _QWORD v6[4]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v7; // [rsp+30h] [rbp-20h]

  v1 = *(__int64 **)(a1 + 16);
  v7 = 261;
  BYTE4(v5) = 0;
  v2 = *v1;
  v6[1] = v1[1];
  v6[0] = v2;
  v3 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v3 )
    sub_B30000((__int64)v3, *(_QWORD *)a1, **(_QWORD ***)(a1 + 8), 0, 0, 0, (__int64)v6, 0, 3, v5, 0);
  return v3;
}

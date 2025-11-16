// Function: sub_2491280
// Address: 0x2491280
//
_QWORD *__fastcall sub_2491280(__int64 a1)
{
  _BYTE *v1; // rax
  _QWORD *v2; // r12
  __int64 v4; // [rsp+8h] [rbp-48h]
  _BYTE *v5; // [rsp+10h] [rbp-40h] BYREF
  __int16 v6; // [rsp+30h] [rbp-20h]

  v1 = *(_BYTE **)(a1 + 16);
  v6 = 257;
  if ( *v1 )
  {
    v5 = v1;
    LOBYTE(v6) = 3;
  }
  BYTE4(v4) = 0;
  v2 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v2 )
    sub_B30000((__int64)v2, *(_QWORD *)a1, *(_QWORD **)(a1 + 8), 0, 0, 0, (__int64)&v5, 0, 3, v4, 0);
  return v2;
}

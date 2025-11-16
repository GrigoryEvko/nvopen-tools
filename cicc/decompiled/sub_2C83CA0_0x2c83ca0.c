// Function: sub_2C83CA0
// Address: 0x2c83ca0
//
char __fastcall sub_2C83CA0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  char result; // al
  __int64 v4; // rbx
  __int64 v6; // rdi
  __int64 i; // [rsp+8h] [rbp-48h]
  char v8; // [rsp+1Eh] [rbp-32h] BYREF
  _BYTE v9[49]; // [rsp+1Fh] [rbp-31h] BYREF

  result = a1 + 48;
  v4 = *(_QWORD *)(a1 + 56);
  for ( i = a1 + 48; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    v6 = v4 - 24;
    if ( !v4 )
      v6 = 0;
    v8 = 0;
    v9[0] = 0;
    result = sub_2C83AE0(v6, &v8, v9);
    *a2 |= v8;
    *a3 |= v9[0];
  }
  return result;
}

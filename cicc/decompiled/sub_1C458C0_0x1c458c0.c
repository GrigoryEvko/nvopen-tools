// Function: sub_1C458C0
// Address: 0x1c458c0
//
char __fastcall sub_1C458C0(__int64 a1, _BYTE *a2, _BYTE *a3)
{
  char result; // al
  __int64 v4; // rbx
  __int64 v6; // rdi
  __int64 i; // [rsp+8h] [rbp-48h]
  char v8; // [rsp+1Eh] [rbp-32h] BYREF
  _BYTE v9[49]; // [rsp+1Fh] [rbp-31h] BYREF

  result = a1 + 40;
  v4 = *(_QWORD *)(a1 + 48);
  for ( i = a1 + 40; i != v4; v4 = *(_QWORD *)(v4 + 8) )
  {
    v6 = v4 - 24;
    if ( !v4 )
      v6 = 0;
    v8 = 0;
    v9[0] = 0;
    result = sub_1C45690(v6, &v8, v9);
    *a2 |= v8;
    *a3 |= v9[0];
  }
  return result;
}

// Function: sub_2700A50
// Address: 0x2700a50
//
void __fastcall sub_2700A50(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  bool v4; // zf
  __int64 i; // r12
  __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v7[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *(_BYTE *)(a2 + 25) == 0;
  v6 = a3;
  v7[0] = a4;
  v7[1] = a1;
  v7[2] = &v6;
  if ( !v4 || *(_QWORD *)(a2 + 40) != *(_QWORD *)(a2 + 32) )
    *a4 = 1;
  if ( !*(_BYTE *)(a2 + 24) )
    sub_26FFED0((__int64)v7, (__int64 **)a2);
  for ( i = *(_QWORD *)(a2 + 104); a2 + 88 != i; i = sub_220EEE0(i) )
  {
    if ( *(_BYTE *)(i + 81) || *(_QWORD *)(i + 96) != *(_QWORD *)(i + 88) )
      *(_BYTE *)v7[0] = 1;
    if ( !*(_BYTE *)(i + 80) )
      sub_26FFED0((__int64)v7, (__int64 **)(i + 56));
  }
}

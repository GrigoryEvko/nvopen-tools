// Function: sub_38A2070
// Address: 0x38a2070
//
__int64 __fastcall sub_38A2070(__int64 a1, _QWORD *a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  unsigned __int64 v8; // r14
  unsigned int v9; // r12d
  __int64 v11; // [rsp+8h] [rbp-48h] BYREF
  __int64 v12[2]; // [rsp+10h] [rbp-40h] BYREF
  char v13; // [rsp+20h] [rbp-30h]
  char v14; // [rsp+21h] [rbp-2Fh]

  v8 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_3891B00(a1, &v11, a3, 0) )
    return 1;
  if ( *(_BYTE *)(v11 + 8) != 8 )
  {
    v9 = sub_38A1070((__int64 **)a1, v11, v12, a4, a5, a6, a7);
    if ( !(_BYTE)v9 )
    {
      *a2 = sub_1624210(v12[0]);
      return v9;
    }
    return 1;
  }
  v14 = 1;
  v13 = 3;
  v12[0] = (__int64)"invalid metadata-value-metadata roundtrip";
  return (unsigned int)sub_38814C0(a1 + 8, v8, (__int64)v12);
}

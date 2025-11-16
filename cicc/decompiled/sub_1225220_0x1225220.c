// Function: sub_1225220
// Address: 0x1225220
//
__int64 __fastcall sub_1225220(__int64 a1, _QWORD *a2, int *a3, __int64 *a4)
{
  unsigned __int64 v6; // r14
  __int64 *v7; // rsi
  unsigned int v8; // r12d
  __int64 *v10; // [rsp+8h] [rbp-58h] BYREF
  __int64 v11[4]; // [rsp+10h] [rbp-50h] BYREF
  char v12; // [rsp+30h] [rbp-30h]
  char v13; // [rsp+31h] [rbp-2Fh]

  v6 = *(_QWORD *)(a1 + 232);
  if ( !(unsigned __int8)sub_12190A0(a1, &v10, a3, 0) )
  {
    v7 = v10;
    if ( *((_BYTE *)v10 + 8) == 9 )
    {
      v13 = 1;
      v12 = 3;
      v11[0] = (__int64)"invalid metadata-value-metadata roundtrip";
      sub_11FD800(a1 + 176, v6, (__int64)v11, 1);
    }
    else
    {
      v8 = sub_1224B80((__int64 **)a1, (__int64)v10, v11, a4);
      if ( !(_BYTE)v8 )
      {
        *a2 = sub_B98A20(v11[0], (__int64)v7);
        return v8;
      }
    }
  }
  return 1;
}

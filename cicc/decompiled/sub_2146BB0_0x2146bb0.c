// Function: sub_2146BB0
// Address: 0x2146bb0
//
__int64 __fastcall sub_2146BB0(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        _DWORD *a4,
        _DWORD *a5,
        const __m128i *a6)
{
  unsigned __int64 v8; // r14
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rax
  char v12; // dl
  __int64 v13; // rax
  bool v14; // al
  _BYTE v16[8]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]

  v8 = sub_2013D30(a1, a2, a3, (__int64)a4, (__int64)a5, a6);
  v10 = v9;
  v11 = *(_QWORD *)(v8 + 40) + 16LL * (unsigned int)v9;
  v12 = *(_BYTE *)v11;
  v13 = *(_QWORD *)(v11 + 8);
  v16[0] = v12;
  v17 = v13;
  if ( v12 )
  {
    if ( (unsigned __int8)(v12 - 14) > 0x5Fu )
    {
      v14 = (unsigned __int8)(v12 - 2) <= 5u;
      goto LABEL_6;
    }
    return sub_2017DE0(a1, v8, v10, a4, a5);
  }
  if ( sub_1F58D20((__int64)v16) )
    return sub_2017DE0(a1, v8, v10, a4, a5);
  v14 = sub_1F58CF0((__int64)v16);
LABEL_6:
  if ( v14 )
    return sub_20174B0(a1, v8, v10, a4, a5);
  else
    return sub_2016B80(a1, v8, v10, a4, a5);
}

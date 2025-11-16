// Function: sub_16A0030
// Address: 0x16a0030
//
__int64 __fastcall sub_16A0030(__int64 a1, __int64 a2)
{
  bool v2; // al
  unsigned int v3; // r12d
  __int16 *v5; // [rsp+0h] [rbp-C0h]
  __int16 *v6[4]; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD v7[4]; // [rsp+30h] [rbp-90h] BYREF
  __int64 v8[5]; // [rsp+50h] [rbp-70h] BYREF
  void *v9[9]; // [rsp+78h] [rbp-48h] BYREF

  v2 = (*(_BYTE *)(a1 + 18) & 6) != 0;
  v3 = v2 && (*(_BYTE *)(a1 + 18) & 7) != 3;
  if ( v2 && (*(_BYTE *)(a1 + 18) & 7) != 3 )
  {
    if ( (unsigned int)sub_1698BD0(a1) == *(_DWORD *)(*(_QWORD *)a1 + 4LL) - 1 )
    {
      sub_1699170((__int64)v6, *(_QWORD *)a1, 1);
      if ( (unsigned int)sub_16994B0(v6, a1, 0) || sub_16984B0((__int64)v6) )
      {
        v3 = 0;
      }
      else if ( a2 )
      {
        v5 = *(__int16 **)a1;
        sub_16986C0(v7, (__int64 *)v6);
        sub_1698450((__int64)v8, (__int64)v7);
        sub_169E320(v9, v8, v5);
        sub_1698460((__int64)v8);
        sub_169ED90((void **)(a2 + 8), v9);
        sub_127D120(v9);
        sub_1698460((__int64)v7);
      }
      sub_1698460((__int64)v6);
    }
    else
    {
      return 0;
    }
  }
  return v3;
}

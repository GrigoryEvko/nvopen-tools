// Function: sub_250C3F0
// Address: 0x250c3f0
//
__int64 __fastcall sub_250C3F0(unsigned __int64 a1, __int64 a2)
{
  unsigned __int8 v3; // dl
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned __int64 v8; // rbx
  __int64 v9; // rdx
  unsigned __int8 v10; // al
  unsigned __int8 v11; // al
  _QWORD v12[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v13[6]; // [rsp+10h] [rbp-30h] BYREF

  if ( a2 == *(_QWORD *)(a1 + 8) )
    return a1;
  v3 = *(_BYTE *)a1;
  if ( *(_BYTE *)a1 == 13 )
    return sub_ACADE0((__int64 **)a2);
  if ( (unsigned int)v3 - 12 <= 1 )
    return sub_ACA8A0((__int64 **)a2);
  if ( v3 > 0x15u )
    return 0;
  if ( sub_AC30F0(a1) )
    return sub_AD6530(a2, a2);
  v5 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v5 + 8) != 14 || *(_BYTE *)(a2 + 8) != 14 )
  {
    v6 = sub_BCAE30(v5);
    v12[1] = v7;
    v12[0] = v6;
    v8 = sub_CA1930(v12);
    v13[0] = sub_BCAE30(a2);
    v13[1] = v9;
    if ( v8 >= sub_CA1930(v13) )
    {
      v10 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 8LL);
      if ( v10 != 12 )
      {
        if ( v10 <= 3u || v10 == 5 || (v10 & 0xFD) == 4 )
        {
          v11 = *(_BYTE *)(a2 + 8);
          if ( v11 <= 3u || v11 == 5 || (v11 & 0xFD) == 4 )
            return sub_AA93C0(0x2Du, a1, a2);
        }
        return 0;
      }
      if ( *(_BYTE *)(a2 + 8) == 12 )
        return sub_AD4C30(a1, (__int64 **)a2, 1);
    }
    return 0;
  }
  return sub_ADAFB0(a1, a2);
}

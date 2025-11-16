// Function: sub_7F6BA0
// Address: 0x7f6ba0
//
__int64 __fastcall sub_7F6BA0(const __m128i *a1)
{
  _BYTE *v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax
  __m128i *v5; // rax
  _BYTE *v6[8]; // [rsp+0h] [rbp-40h] BYREF

  if ( (unsigned int)sub_8D3B80(a1) || (unsigned int)sub_8D2B80(a1) )
  {
    v1 = sub_724D50(10);
    *((_QWORD *)v1 + 16) = a1;
    v2 = (__int64)v1;
    sub_7F5D80((__int64)a1, 11, (__int64)v6);
    if ( v6[3] )
    {
      v3 = sub_7F6BA0();
      *(_QWORD *)(v2 + 184) = v3;
      *(_QWORD *)(v2 + 176) = v3;
    }
  }
  else
  {
    v6[0] = sub_724DC0();
    v5 = sub_73D720(a1);
    sub_72BB40((__int64)v5, (const __m128i *)v6[0]);
    v2 = sub_724E50((__int64 *)v6, v6[0]);
  }
  *(_BYTE *)(v2 - 8) &= ~8u;
  *(_BYTE *)(v2 + 171) |= 0x40u;
  return v2;
}

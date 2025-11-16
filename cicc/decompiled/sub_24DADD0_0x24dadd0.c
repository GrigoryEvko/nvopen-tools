// Function: sub_24DADD0
// Address: 0x24dadd0
//
void __fastcall sub_24DADD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // r14
  __int64 v5; // rsi
  __int64 v6; // rdi
  unsigned int v7; // [rsp-4Ch] [rbp-4Ch] BYREF
  __m128i v8; // [rsp-48h] [rbp-48h] BYREF
  __int64 v9; // [rsp-38h] [rbp-38h]

  if ( LOBYTE(qword_4FEC628[8]) )
  {
    v3 = *(_QWORD *)(a2 - 32);
    if ( v3 )
    {
      if ( !*(_BYTE *)v3 && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a2 + 80) )
      {
        if ( (v4 = *(__int64 **)(a1 + 8), !(unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 23))
          && !(unsigned __int8)sub_B49560(a2, 23)
          || (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 4)
          || (unsigned __int8)sub_B49560(a2, 4) )
        {
          v5 = *(_QWORD *)(a2 - 32);
          if ( v5
            && !*(_BYTE *)v5
            && *(_QWORD *)(v5 + 24) == *(_QWORD *)(a2 + 80)
            && sub_981210(*v4, v5, &v7)
            && (v7 == 357 || v7 == 186)
            && **(_BYTE **)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))) != 17 )
          {
            v6 = *(_QWORD *)(a1 + 16);
            v8.m128i_i64[0] = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
            v8.m128i_i64[1] = a2;
            v9 = a2;
            sub_24DAD90(v6, &v8);
          }
        }
      }
    }
  }
}

// Function: sub_253A2E0
// Address: 0x253a2e0
//
__int64 __fastcall sub_253A2E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  char v3; // [rsp+17h] [rbp-41h] BYREF
  unsigned int v4; // [rsp+18h] [rbp-40h] BYREF
  __int64 v5; // [rsp+1Ch] [rbp-3Ch] BYREF
  int v6; // [rsp+24h] [rbp-34h]
  _QWORD v7[5]; // [rsp+28h] [rbp-30h] BYREF

  v7[2] = &v4;
  v5 = 0xB00000005LL;
  v7[0] = a2;
  v4 = 1;
  v7[1] = a1;
  v3 = 0;
  v6 = 56;
  if ( (unsigned __int8)sub_2526370(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_25A0950,
                          (__int64)v7,
                          a1,
                          (int *)&v5,
                          3,
                          &v3,
                          1,
                          0) )
    return v4;
  if ( !*(_BYTE *)(a1 + 168) )
    v4 = 0;
  result = 0;
  if ( *(_BYTE *)(a1 + 169) )
    result = v4;
  *(_WORD *)(a1 + 168) = 257;
  return result;
}

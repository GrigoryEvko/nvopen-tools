// Function: sub_253A550
// Address: 0x253a550
//
__int64 __fastcall sub_253A550(__int64 a1, __int64 a2)
{
  char v3; // [rsp+3h] [rbp-4Dh] BYREF
  __int64 v4; // [rsp+4h] [rbp-4Ch] BYREF
  int v5; // [rsp+Ch] [rbp-44h]
  _QWORD v6[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v7[6]; // [rsp+20h] [rbp-30h] BYREF

  v6[0] = a2;
  v6[1] = a1;
  v7[0] = a2;
  v7[1] = a1;
  v3 = 0;
  if ( (unsigned __int8)sub_25264B0(
                          a2,
                          (unsigned __int8 (__fastcall *)(__int64, unsigned __int64, __int64))sub_2535840,
                          (__int64)v6,
                          a1,
                          &v3) )
  {
    v4 = 0xB00000005LL;
    v5 = 56;
    if ( (unsigned __int8)sub_2526370(
                            a2,
                            (__int64 (__fastcall *)(__int64, unsigned __int64, __int64))sub_259BAE0,
                            (__int64)v7,
                            a1,
                            (int *)&v4,
                            3,
                            &v3,
                            0,
                            0) )
      return 1;
  }
  *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
  return 0;
}

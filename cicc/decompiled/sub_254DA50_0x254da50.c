// Function: sub_254DA50
// Address: 0x254da50
//
__int64 __fastcall sub_254DA50(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // r8
  char v4; // dl
  unsigned __int64 v5; // r8
  unsigned int v6; // eax
  unsigned int v7; // eax
  __int64 v9; // rdx
  __int64 v10; // rdx
  char v11; // [rsp+Bh] [rbp-45h] BYREF
  unsigned int v12; // [rsp+Ch] [rbp-44h] BYREF
  _QWORD v13[8]; // [rsp+10h] [rbp-40h] BYREF

  v2 = sub_2509800((_QWORD *)(a1 + 72));
  v3 = *(_QWORD *)(a1 + 72);
  v11 = v2;
  v4 = v3;
  v5 = v3 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (v4 & 3) == 3 )
    v5 = *(_QWORD *)(v5 + 24);
  v13[1] = v5;
  v13[0] = &v11;
  v13[4] = &v12;
  v12 = 1;
  v13[2] = a2;
  v13[3] = a1;
  v13[5] = a1 + 88;
  if ( (unsigned __int8)sub_251CC40(
                          a2,
                          (__int64 (__fastcall *)(__int64, unsigned __int64 *, __int64))sub_258A130,
                          (__int64)v13,
                          a1,
                          v5) )
    return v12;
  if ( *(_DWORD *)(a1 + 112) <= 0x40u && (v6 = *(_DWORD *)(a1 + 144), v6 <= 0x40) )
  {
    v10 = *(_QWORD *)(a1 + 136);
    *(_DWORD *)(a1 + 112) = v6;
    *(_QWORD *)(a1 + 104) = v10;
  }
  else
  {
    sub_C43990(a1 + 104, a1 + 136);
  }
  if ( *(_DWORD *)(a1 + 128) <= 0x40u && (v7 = *(_DWORD *)(a1 + 160), v7 <= 0x40) )
  {
    v9 = *(_QWORD *)(a1 + 152);
    *(_DWORD *)(a1 + 128) = v7;
    *(_QWORD *)(a1 + 120) = v9;
    return 0;
  }
  else
  {
    sub_C43990(a1 + 120, a1 + 152);
    return 0;
  }
}

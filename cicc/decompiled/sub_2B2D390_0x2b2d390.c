// Function: sub_2B2D390
// Address: 0x2b2d390
//
__int64 __fastcall sub_2B2D390(__int64 a1, __int64 a2)
{
  int v3; // edx
  __int64 ***v4; // rax
  __int64 **v5; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int8 *v12; // rsi
  int v13; // edx
  int v14; // eax
  __int64 v15[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v3 - 17) > 1 )
  {
    if ( (_BYTE)v3 != 14 )
      return sub_AD62B0(a2);
    v5 = (__int64 **)a2;
  }
  else
  {
    v4 = *(__int64 ****)(a2 + 16);
    v5 = *v4;
    if ( *((_BYTE *)*v4 + 8) != 14 )
      return sub_AD62B0(a2);
    if ( (unsigned __int8)(v3 - 17) >= 2u )
      v5 = (__int64 **)a2;
  }
  v7 = sub_9208B0(a1, (__int64)v5);
  v15[1] = v8;
  v15[0] = (v7 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v9 = sub_CA1930(v15);
  v10 = sub_BCCE00(*(_QWORD **)a2, v9);
  v11 = sub_AD62B0(v10);
  v12 = (unsigned __int8 *)sub_AD4C70(v11, v5, 0);
  v13 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned int)(v13 - 17) > 1 )
    return (__int64)v12;
  v14 = *(_DWORD *)(a2 + 32);
  BYTE4(v15[0]) = (_BYTE)v13 == 18;
  LODWORD(v15[0]) = v14;
  return sub_AD5E10(v15[0], v12);
}

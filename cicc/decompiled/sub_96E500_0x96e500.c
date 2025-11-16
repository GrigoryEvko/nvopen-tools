// Function: sub_96E500
// Address: 0x96e500
//
__int64 __fastcall sub_96E500(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 v7; // rsi
  unsigned __int64 v8; // rbx
  char v9; // dl
  char v10; // dl
  int v11; // edx
  unsigned __int8 v12; // al
  unsigned __int8 v13; // cl
  char v14; // [rsp+Fh] [rbp-41h]

  v4 = *a1;
  if ( (_BYTE)v4 == 13 )
    return sub_ACADE0(a2);
  if ( (unsigned int)(v4 - 12) <= 1 )
    return sub_ACA8A0(a2);
  v7 = *((_QWORD *)a1 + 1);
  v8 = (sub_9208B0(a3, v7) + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v14 = v9;
  if ( sub_9208B0(a3, v7) != v8 || v10 != v14 )
    return 0;
  if ( (unsigned __int8)sub_AC30F0(a1) && *(_BYTE *)(a2 + 8) != 10 )
    return sub_AD6530(a2);
  if ( !(unsigned __int8)sub_AD7930(a1) )
    return 0;
  v11 = *(unsigned __int8 *)(a2 + 8);
  v12 = *(_BYTE *)(a2 + 8);
  if ( (unsigned int)(v11 - 17) > 1 )
  {
    if ( (_BYTE)v11 == 12 )
      return sub_AD62B0(a2);
  }
  else
  {
    v13 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
    if ( v13 == 12 )
      return sub_AD62B0(a2);
    if ( v11 == 18 )
      goto LABEL_15;
  }
  if ( v11 != 17 )
    goto LABEL_16;
  v13 = *(_BYTE *)(**(_QWORD **)(a2 + 16) + 8LL);
LABEL_15:
  v12 = v13;
LABEL_16:
  if ( v12 > 3u && v12 != 5 && (v12 & 0xFD) != 4 )
    return 0;
  return sub_AD62B0(a2);
}

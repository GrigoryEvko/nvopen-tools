// Function: sub_3035DC0
// Address: 0x3035dc0
//
__int64 __fastcall sub_3035DC0(__int64 a1, unsigned __int16 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  unsigned int v5; // ebx
  unsigned __int16 v7; // ax
  __int64 v8; // rdx
  __int64 v9; // rdx
  char v10; // al
  unsigned int v11; // r13d
  __int16 v12; // ax
  __int64 v13; // rax
  unsigned __int16 v14; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  char v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]
  __int64 v19; // [rsp+28h] [rbp-28h]

  if ( (unsigned __int16)(a2 - 17) > 0xD3u )
    goto LABEL_5;
  v15 = 0;
  v4 = a2 - 1;
  v14 = word_4456580[v4];
  if ( (unsigned __int16)(a2 - 176) <= 0x34u )
  {
    sub_CA17B0(
      "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use EVT::"
      "getVectorElementCount() instead");
    sub_CA17B0(
      "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use MVT::"
      "getVectorElementCount() instead");
LABEL_5:
    *(_BYTE *)(a1 + 24) = 0;
    return a1;
  }
  v5 = word_4456340[v4];
  switch ( a2 )
  {
    case 0x23u:
    case 0x25u:
    case 0x2Fu:
    case 0x31u:
    case 0x3Au:
    case 0x3Cu:
    case 0x74u:
    case 0x7Fu:
    case 0x81u:
    case 0x8Au:
    case 0x8Cu:
    case 0x93u:
    case 0x95u:
    case 0xA7u:
      goto LABEL_8;
    case 0x26u:
    case 0x27u:
    case 0x32u:
    case 0x82u:
    case 0x8Du:
      goto LABEL_10;
    case 0x28u:
    case 0x33u:
    case 0x83u:
    case 0x8Eu:
      if ( !a4 )
        goto LABEL_5;
LABEL_10:
      if ( v14 )
      {
        if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
          BUG();
        v13 = 16LL * (v14 - 1);
        v9 = *(_QWORD *)&byte_444C4A0[v13];
        v10 = byte_444C4A0[v13 + 8];
      }
      else
      {
        v18 = sub_3007260((__int64)&v14);
        v19 = v8;
        v9 = v18;
        v10 = v19;
      }
      v16 = v9;
      v17 = v10;
      v11 = 0x20uLL / sub_CA1930(&v16);
      v12 = sub_2D43050(v14, v11);
      *(_BYTE *)(a1 + 24) = 1;
      *(_QWORD *)(a1 + 16) = 0;
      *(_WORD *)(a1 + 8) = v12;
      *(_DWORD *)a1 = v5 / v11;
      break;
    case 0x40u:
    case 0x76u:
    case 0x99u:
    case 0xA9u:
      if ( !a4 )
        goto LABEL_5;
LABEL_8:
      v7 = v14;
      *(_DWORD *)a1 = v5;
      *(_BYTE *)(a1 + 24) = 1;
      *(_WORD *)(a1 + 8) = v7;
      *(_QWORD *)(a1 + 16) = v15;
      break;
    default:
      goto LABEL_5;
  }
  return a1;
}

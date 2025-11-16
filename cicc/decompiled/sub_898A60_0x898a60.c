// Function: sub_898A60
// Address: 0x898a60
//
_QWORD *__fastcall sub_898A60(
        int a1,
        int a2,
        __int64 a3,
        int a4,
        int a5,
        int a6,
        char a7,
        __int64 a8,
        __int64 a9,
        __int64 a10)
{
  __int64 *v13; // r14
  _QWORD *result; // rax
  char v15; // dl

  v13 = sub_8978E0(a1, a2, a3, a4, a8, a9);
  result = sub_880AD0((__int64)v13);
  if ( a5 )
  {
    v15 = *((_BYTE *)result + 56);
    *((_BYTE *)result + 56) = v15 | 0x40;
    if ( !a4 )
      return result;
    if ( !a6 )
      *((_BYTE *)result + 56) = v15 | 0x50;
    *((_BYTE *)v13 + 84) |= 0x20u;
    *((_BYTE *)result + 56) |= 0x40u;
LABEL_6:
    *(_QWORD *)(a10 + 84) = 0x100000001LL;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 7) |= 1u;
    *((_BYTE *)result + 56) = (32 * (a7 & 1)) | result[7] & 0xDF;
    *((_BYTE *)v13 + 84) = *((_BYTE *)v13 + 84) & 0xBF | ((a7 & 1) << 6);
    return result;
  }
  if ( a4 )
  {
    *((_BYTE *)result + 56) |= 0x10u;
    goto LABEL_6;
  }
  return result;
}

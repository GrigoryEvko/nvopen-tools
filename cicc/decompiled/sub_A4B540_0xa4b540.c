// Function: sub_A4B540
// Address: 0xa4b540
//
__int64 __fastcall sub_A4B540(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v4; // al
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  char v9; // [rsp+8h] [rbp-18h]

  v4 = (*(_BYTE *)(a3 + 8) >> 1) & 7;
  switch ( v4 )
  {
    case 2:
      sub_A4B2C0(a1, a2, *(_QWORD *)a3, a4);
      return a1;
    case 4:
      sub_9C66D0((__int64)&v8, a2, 6, a4);
      if ( (v9 & 1) != 0 )
      {
        v7 = v8;
        *(_BYTE *)(a1 + 8) |= 3u;
        *(_QWORD *)a1 = v7 & 0xFFFFFFFFFFFFFFFELL;
      }
      else
      {
        v5 = (unsigned int)v8;
        *(_BYTE *)(a1 + 8) = *(_BYTE *)(a1 + 8) & 0xFC | 2;
        *(_QWORD *)a1 = aAbcdefghijklmn[v5];
      }
      return a1;
    case 1:
      sub_9C66D0(a1, a2, *(_QWORD *)a3, a4);
      return a1;
    default:
      BUG();
  }
}

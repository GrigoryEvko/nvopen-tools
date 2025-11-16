// Function: sub_2D49200
// Address: 0x2d49200
//
__int64 __fastcall sub_2D49200(unsigned __int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v5; // rdi
  __int64 (*v6)(); // rax
  _BYTE v8[9]; // [rsp+1Fh] [rbp-11h] BYREF

  v5 = *a1;
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 1144LL);
  if ( v6 != sub_2D42A80 )
  {
    switch ( ((unsigned int (__fastcall *)(unsigned __int64, __int64, __int64, __int64, _QWORD))v6)(v5, a2, a3, a4, 0) )
    {
      case 0u:
        return 0;
      case 2u:
        sub_2D48EC0(
          a1,
          (_QWORD *)a2,
          *(_QWORD *)(a2 + 8),
          *(_QWORD *)(a2 - 32),
          (*(_WORD *)(a2 + 2) >> 7) & 7,
          *(_QWORD *)(a2 + 8),
          sub_2D42AE0,
          (__int64)v8);
        return 1;
      case 3u:
        return sub_2D47BB0(a1, a2);
      case 4u:
        return sub_2D48FA0(a1[1], a2);
      case 9u:
        *(_WORD *)(a2 + 2) &= 0xFC7Fu;
        *(_BYTE *)(a2 + 72) = 1;
        return 1;
      default:
        BUG();
    }
  }
  return 0;
}

// Function: sub_36D8E00
// Address: 0x36d8e00
//
__int64 __fastcall sub_36D8E00(__int64 a1, __int64 a2, int a3)
{
  _BYTE *v4; // rax

  sub_3420B80(a1, a2, a3);
  *(_QWORD *)a1 = off_4A3BE98;
  *(_QWORD *)(a1 + 952) = a2;
  *(_BYTE *)(a1 + 961) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  memset((void *)(a1 + 976), 0, 0xA0u);
  *(_BYTE *)(a1 + 984) = 1;
  v4 = (_BYTE *)(a1 + 992);
  *(_DWORD *)(a1 + 988) = 0;
  *(_DWORD *)(a1 + 984) &= 1u;
  do
  {
    if ( v4 )
      *v4 = -1;
    v4 += 8;
  }
  while ( (_BYTE *)(a1 + 1056) != v4 );
  *(_QWORD *)(a1 + 1136) = 0;
  *(_QWORD *)(a1 + 1056) = a1 + 1072;
  *(_QWORD *)(a1 + 1064) = 0x800000000LL;
  *(_BYTE *)(a1 + 960) = a3 > 0;
  return 0x800000000LL;
}

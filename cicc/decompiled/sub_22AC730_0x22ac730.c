// Function: sub_22AC730
// Address: 0x22ac730
//
__int64 __fastcall sub_22AC730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 i; // rbx

  *(_QWORD *)(a1 + 48) = a1 + 72;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 208) = a1 + 200;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 200) = (a1 + 200) | 4;
  *(_QWORD *)(a1 + 224) = a1 + 248;
  *(_QWORD *)(a1 + 16) = a4;
  *(_QWORD *)(a1 + 24) = a5;
  *(_QWORD *)(a1 + 32) = a6;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 16;
  *(_DWORD *)(a1 + 64) = 0;
  *(_BYTE *)(a1 + 68) = 1;
  *(_QWORD *)(a1 + 232) = 32;
  *(_DWORD *)(a1 + 240) = 0;
  *(_BYTE *)(a1 + 244) = 1;
  *(_QWORD *)(a1 + 216) = 1;
  sub_30AB790(a2, a3, a1 + 216);
  result = **(_QWORD **)(a2 + 32);
  for ( i = *(_QWORD *)(result + 56); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) != 84 )
      break;
    result = sub_22AC0D0(a1, (unsigned __int8 *)(i - 24));
  }
  return result;
}

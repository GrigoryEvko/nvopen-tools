// Function: sub_9C6430
// Address: 0x9c6430
//
__int64 __fastcall sub_9C6430(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  unsigned __int64 v5; // rax
  __int64 v6; // rdx

  v5 = *(unsigned __int8 *)(a1 + 8);
  if ( (unsigned __int8)v5 <= 0xCu )
  {
    v6 = 4143;
    a5 = 1;
    if ( _bittest64(&v6, v5) )
      return a5;
  }
  LOBYTE(a5) = (v5 & 0xFB) == 10 || (v5 & 0xFD) == 4;
  if ( (_BYTE)a5 || (unsigned __int8)(v5 - 15) > 3u && (_BYTE)v5 != 20 )
    return a5;
  else
    return sub_BCEBA0(a1, a2);
}

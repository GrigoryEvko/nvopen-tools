// Function: sub_AF2660
// Address: 0xaf2660
//
__int64 __fastcall sub_AF2660(unsigned __int8 *a1)
{
  unsigned __int64 v1; // rax
  __int64 v2; // rdx
  __int64 v3; // r8
  unsigned __int8 v5; // al
  __int64 v6; // rdi

  v1 = *a1;
  if ( (unsigned __int8)v1 > 0x24u || (v2 = 0x140000F000LL, !_bittest64(&v2, v1)) )
  {
    if ( (_BYTE)v1 != 18 && (unsigned int)(unsigned __int8)v1 - 19 > 1 && (_BYTE)v1 != 21 )
    {
      if ( (_BYTE)v1 == 33 )
        return *(_QWORD *)sub_A17150(a1 - 16);
      v3 = 0;
      if ( (_BYTE)v1 != 22 )
        return v3;
    }
    return *((_QWORD *)sub_A17150(a1 - 16) + 1);
  }
  v5 = *(a1 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *((_QWORD *)a1 - 4);
  else
    v6 = (__int64)&a1[-8 * ((v5 >> 2) & 0xF) - 16];
  return *(_QWORD *)(v6 + 8);
}

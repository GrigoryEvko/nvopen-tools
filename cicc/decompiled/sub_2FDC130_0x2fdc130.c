// Function: sub_2FDC130
// Address: 0x2fdc130
//
__int64 __fastcall sub_2FDC130(__int64 a1)
{
  unsigned int v1; // r13d
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned __int8 *v5; // rdi
  int v6; // eax
  unsigned __int64 v7; // rax

  if ( (*(_BYTE *)(a1 + 32) & 0xFu) - 7 <= 1 && !(unsigned __int8)sub_B2DDD0(a1, 0, 0, 1, 0, 0, 0) )
  {
    v1 = sub_B2D610(a1, 34);
    if ( (_BYTE)v1 )
    {
      v3 = 0x8000000000041LL;
      v4 = *(_QWORD *)(a1 + 16);
      if ( !v4 )
        return v1;
      while ( 1 )
      {
        v5 = *(unsigned __int8 **)(v4 + 24);
        v6 = *v5;
        if ( (unsigned __int8)v6 > 0x1Cu )
        {
          v7 = (unsigned int)(v6 - 34);
          if ( (unsigned __int8)v7 <= 0x33u && _bittest64(&v3, v7) && sub_B49220((__int64)v5) )
            break;
        }
        v4 = *(_QWORD *)(v4 + 8);
        if ( !v4 )
          return v1;
      }
    }
  }
  return 0;
}

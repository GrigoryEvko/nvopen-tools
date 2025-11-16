// Function: malloc
// Address: 0x1305140
//
// Alternative name is '__libc_malloc'
__int64 __fastcall malloc(unsigned __int64 a1)
{
  __int64 v1; // rax
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rax
  __int64 *v4; // rsi
  __int64 v5; // r9
  _QWORD *v6; // r8

  if ( a1 <= 0x1000 )
  {
    v1 = byte_5060800[(a1 + 7) >> 3];
    v2 = qword_505FA40[v1] + __readfsqword(0xFFFFF8D0);
    if ( __readfsqword(0xFFFFF8D8) > v2 )
    {
      v3 = __readfsqword(0) + 24 * v1 - 2664;
      v4 = *(__int64 **)(v3 + 864);
      v5 = *v4;
      v6 = v4 + 1;
      if ( (_WORD)v4 != *(_WORD *)(v3 + 880) )
      {
        *(_QWORD *)(v3 + 864) = v6;
        __writefsqword(0xFFFFF8D0, v2);
        ++*(_QWORD *)(v3 + 872);
        return v5;
      }
      if ( (_WORD)v4 != *(_WORD *)(v3 + 884) )
      {
        *(_QWORD *)(v3 + 864) = v6;
        *(_WORD *)(v3 + 880) = (_WORD)v6;
        __writefsqword(0xFFFFF8D0, v2);
        ++*(_QWORD *)(v3 + 872);
        return v5;
      }
    }
  }
  return sub_13047E0(a1);
}

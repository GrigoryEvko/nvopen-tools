// Function: sub_D92AA0
// Address: 0xd92aa0
//
char __fastcall sub_D92AA0(__int64 a1, __int64 a2)
{
  __int16 v2; // dx
  __int64 v3; // r12
  unsigned int v4; // r13d
  int v5; // r8d
  char result; // al
  int v8; // ebx
  unsigned __int64 v9; // rax

  v2 = *(_WORD *)(a2 + 24);
  if ( v2 )
  {
    result = 0;
    if ( v2 == 1 )
      return sub_B2D610(**(_QWORD **)a1, 96);
  }
  else
  {
    v3 = *(_QWORD *)(a2 + 32);
    v4 = *(_DWORD *)(v3 + 32);
    if ( v4 <= 0x40 )
    {
      _RDX = *(_QWORD *)(v3 + 24);
      result = 0;
      if ( _RDX )
      {
        result = 1;
        if ( (_RDX & (_RDX - 1)) != 0 )
        {
          result = *(_BYTE *)(a1 + 8);
          if ( result )
          {
            if ( _bittest64(&_RDX, v4 - 1) )
            {
              if ( v4 )
              {
                v8 = 64;
                if ( _RDX << (64 - (unsigned __int8)v4) != -1 )
                {
                  _BitScanReverse64(&v9, ~(_RDX << (64 - (unsigned __int8)v4)));
                  v8 = v9 ^ 0x3F;
                }
              }
              else
              {
                v8 = 0;
              }
              __asm { tzcnt   rax, rdx }
              if ( (unsigned int)_RAX > v4 )
                LODWORD(_RAX) = *(_DWORD *)(v3 + 32);
              return v8 + (_DWORD)_RAX == v4;
            }
            return 0;
          }
        }
      }
    }
    else
    {
      v5 = sub_C44630(v3 + 24);
      result = 1;
      if ( v5 != 1 )
      {
        result = *(_BYTE *)(a1 + 8);
        if ( result )
        {
          result = 0;
          if ( (*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8LL * ((v4 - 1) >> 6)) & (1LL << ((unsigned __int8)v4 - 1))) != 0 )
          {
            v8 = sub_C44500(v3 + 24);
            LODWORD(_RAX) = sub_C44590(v3 + 24);
            return v8 + (_DWORD)_RAX == v4;
          }
        }
      }
    }
  }
  return result;
}

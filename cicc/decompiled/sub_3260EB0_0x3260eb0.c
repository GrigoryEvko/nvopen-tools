// Function: sub_3260EB0
// Address: 0x3260eb0
//
bool __fastcall sub_3260EB0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int16 v3; // r14
  unsigned int v4; // r12d
  int v5; // r8d
  bool result; // al
  int v7; // r8d
  int v9; // ebx
  unsigned __int64 v10; // rax
  int v12; // eax
  __int64 v13; // rdi

  v2 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v3 = *(_WORD *)(*(_QWORD *)a2 + 32LL);
  v4 = *(_DWORD *)(v2 + 32);
  if ( v4 <= 0x40 )
  {
    result = 0;
    if ( *(_QWORD *)(v2 + 24) )
    {
      if ( (v3 & 8) == 0 )
      {
        _RDX = *(_QWORD *)(v2 + 24);
        if ( _RDX )
        {
          result = 1;
          if ( (_RDX & (_RDX - 1)) != 0 )
          {
            if ( _bittest64(&_RDX, v4 - 1) )
            {
              if ( v4 )
              {
                v9 = 64;
                if ( _RDX << (64 - (unsigned __int8)v4) != -1 )
                {
                  _BitScanReverse64(&v10, ~(_RDX << (64 - (unsigned __int8)v4)));
                  v9 = v10 ^ 0x3F;
                }
              }
              else
              {
                v9 = 0;
              }
              __asm { tzcnt   rdx, rdx }
              v12 = _RDX;
              if ( (unsigned int)_RDX > v4 )
                v12 = v4;
              return v9 + v12 == v4;
            }
            return 0;
          }
        }
      }
    }
  }
  else
  {
    v5 = sub_C444A0(v2 + 24);
    result = 0;
    if ( v4 != v5 && (v3 & 8) == 0 )
    {
      v7 = sub_C44630(v2 + 24);
      result = 1;
      if ( v7 != 1 )
      {
        result = 0;
        if ( (*(_QWORD *)(*(_QWORD *)(v2 + 24) + 8LL * ((v4 - 1) >> 6)) & (1LL << ((unsigned __int8)v4 - 1))) != 0 )
        {
          v13 = v2 + 24;
          v9 = sub_C44500(v2 + 24);
          v12 = sub_C44590(v13);
          return v9 + v12 == v4;
        }
      }
    }
  }
  return result;
}

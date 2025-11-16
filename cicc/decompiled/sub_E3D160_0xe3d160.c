// Function: sub_E3D160
// Address: 0xe3d160
//
unsigned __int64 __fastcall sub_E3D160(unsigned __int64 _RDI, unsigned __int64 a2, int a3)
{
  unsigned __int64 v3; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v6; // rdx
  int v7; // ecx
  unsigned __int64 v8; // rdx
  unsigned int v9; // eax
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  __int64 v13; // [rsp+0h] [rbp-10h]

  switch ( a3 )
  {
    case 27:
      if ( !a2 )
        return v13;
      return _RDI / a2;
    case 28:
      if ( a2 <= _RDI )
        return _RDI - a2;
      return v13;
    case 30:
      if ( !_RDI )
        return _RDI * a2;
      _BitScanReverse64(&v6, _RDI);
      v7 = 63 - (v6 ^ 0x3F);
      if ( !a2 )
        return _RDI * a2;
      _BitScanReverse64(&v8, a2);
      v9 = v7 + 63 - (v8 ^ 0x3F);
      if ( v9 <= 0x3E )
        return _RDI * a2;
      if ( v9 != 63 )
        return v13;
      v10 = a2 * (_RDI >> 1);
      if ( v10 < 0 )
        return v13;
      result = 2 * v10;
      if ( (_RDI & 1) != 0 )
      {
        v11 = a2 + result;
        if ( a2 < result )
          a2 = result;
        result = v11;
        if ( v11 < a2 )
          return v13;
      }
      return result;
    case 34:
      v12 = _RDI;
      if ( a2 >= _RDI )
        v12 = a2;
      if ( a2 + _RDI >= v12 )
        return a2 + _RDI;
      return v13;
    case 36:
      if ( a2 <= 0x3F )
      {
        if ( !_RDI )
          return _RDI << a2;
        _BitScanReverse64(&v3, _RDI);
        if ( (int)(v3 ^ 0x3F) >= a2 )
          return _RDI << a2;
      }
      return v13;
    case 37:
      if ( a2 <= 0x3F )
      {
        if ( !_RDI )
          return _RDI >> a2;
        __asm { tzcnt   rax, rdi }
        if ( (int)_RAX >= a2 )
          return _RDI >> a2;
      }
      return v13;
    default:
      return v13;
  }
}

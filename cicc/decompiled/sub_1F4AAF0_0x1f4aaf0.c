// Function: sub_1F4AAF0
// Address: 0x1f4aaf0
//
_QWORD *__fastcall sub_1F4AAF0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // r9
  __int64 v5; // r10
  unsigned int *v6; // rdi
  __int64 v7; // r9
  unsigned int v8; // eax
  __int64 v9; // r9
  unsigned int v10; // r8d
  unsigned int *v11; // rdx
  __int64 v14; // rdx
  unsigned int v16; // r11d
  unsigned int *v17; // rdx

  v3 = a2;
  if ( !a2 || *(_BYTE *)(*a2 + 29LL) )
    return v3;
  v5 = *(_QWORD *)(a1 + 256);
  v6 = (unsigned int *)a2[1];
  v7 = *(_QWORD *)(a1 + 264) - v5;
  v8 = *v6;
  v9 = v7 >> 3;
  v10 = v9;
  if ( *v6 )
  {
    _ESI = *v6;
    v8 = 0;
LABEL_10:
    __asm { tzcnt   ecx, esi }
    v14 = _ECX + v8;
    _ESI = _ESI >> _ECX >> 1;
    v16 = v14 + 1;
    if ( (_DWORD)v9 != (_DWORD)v14 )
    {
      while ( 1 )
      {
        v3 = *(_QWORD **)(v5 + 8 * v14);
        if ( *(_BYTE *)(*v3 + 29LL) )
          return v3;
        if ( !_ESI )
        {
          v17 = v6 + 1;
          while ( 1 )
          {
            v8 += 32;
            if ( v10 <= v8 )
              return 0;
            v6 = v17;
            _ESI = *v17++;
            if ( _ESI )
            {
              v16 = v8;
              break;
            }
          }
        }
        __asm { tzcnt   ecx, esi }
        v14 = _ECX + v16;
        _ESI = _ESI >> _ECX >> 1;
        v16 = v14 + 1;
        if ( v10 == (_DWORD)v14 )
          return 0;
      }
    }
  }
  else
  {
    v11 = v6 + 1;
    while ( 1 )
    {
      v8 += 32;
      if ( (unsigned int)v9 <= v8 )
        break;
      v6 = v11;
      _ESI = *v11++;
      if ( _ESI )
        goto LABEL_10;
    }
  }
  return 0;
}

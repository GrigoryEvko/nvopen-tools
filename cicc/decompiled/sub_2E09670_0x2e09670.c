// Function: sub_2E09670
// Address: 0x2e09670
//
_QWORD *__fastcall sub_2E09670(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rsi
  _QWORD *v5; // r8
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 *v8; // rsi

  v3 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( v3 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v7 = v6 >> 1;
        v8 = &v5[(v6 >> 1) + (v6 & 0xFFFFFFFFFFFFFFFELL)];
        if ( (*(_DWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*a3 >> 1) & 3) < (*(_DWORD *)((*v8 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*v8 >> 1)
                                                                                              & 3) )
          break;
        v5 = v8 + 3;
        v6 = v6 - v7 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v6 >>= 1;
    }
    while ( v7 > 0 );
  }
  return v5;
}

// Function: sub_3700E20
// Address: 0x3700e20
//
_QWORD *__fastcall sub_3700E20(_QWORD *a1, __int64 a2)
{
  void (__fastcall ***v3)(_QWORD, _BYTE *, __int64); // rdi
  int v5; // r13d
  _BYTE v6[33]; // [rsp+Fh] [rbp-21h] BYREF

  v3 = *(void (__fastcall ****)(_QWORD, _BYTE *, __int64))(a2 + 56);
  --*(_DWORD *)(a2 + 8);
  if ( v3 )
  {
    if ( !*(_QWORD *)(a2 + 40) && !*(_QWORD *)(a2 + 48) && (*(_DWORD *)(a2 + 64) & 3) != 0 )
    {
      v5 = 4 - (*(_DWORD *)(a2 + 64) & 3);
      while ( 1 )
      {
        v6[0] = v5 - 16;
        (**v3)(v3, v6, 1);
        if ( !--v5 )
          break;
        v3 = *(void (__fastcall ****)(_QWORD, _BYTE *, __int64))(a2 + 56);
      }
      if ( *(_QWORD *)(a2 + 56) && !*(_QWORD *)(a2 + 40) && !*(_QWORD *)(a2 + 48) )
        *(_QWORD *)(a2 + 64) = 4;
    }
  }
  *a1 = 1;
  return a1;
}

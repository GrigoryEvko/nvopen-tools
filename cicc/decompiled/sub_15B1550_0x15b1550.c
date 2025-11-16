// Function: sub_15B1550
// Address: 0x15b1550
//
__int64 __fastcall sub_15B1550(__int64 a1, __int64 a2, unsigned int a3)
{
  _QWORD *v3; // rcx
  __int64 v4; // rax

  v3 = *(_QWORD **)(a1 + 24);
  v4 = (__int64)(*(_QWORD *)(a1 + 32) - (_QWORD)v3) >> 3;
  LOBYTE(a3) = (_DWORD)v4 != 6 && (_DWORD)v4 != 3;
  if ( (_BYTE)a3 )
    return 0;
  if ( *v3 == 16 && v3[2] == 159 )
  {
    a3 = 1;
    if ( (_DWORD)v4 == 6 )
      LOBYTE(a3) = v3[3] == 4096;
  }
  return a3;
}

// Function: sub_28E9340
// Address: 0x28e9340
//
__int64 __fastcall sub_28E9340(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, char a5, _BYTE *a6)
{
  __int64 v9; // rcx
  int v11; // edx
  __int64 v12; // rax
  unsigned __int8 *v13; // r13
  __int64 v15; // rcx

  v9 = *(_QWORD *)(a1 + 8);
  v11 = *(unsigned __int8 *)(v9 + 8);
  if ( (unsigned int)(v11 - 17) <= 1 )
    LOBYTE(v11) = *(_BYTE *)(**(_QWORD **)(v9 + 16) + 8LL);
  if ( (_BYTE)v11 == 12 )
  {
    v15 = a4;
    BYTE1(v15) = a5;
    return sub_B50550(a1, a2, a3, v15);
  }
  else
  {
    v12 = a4;
    BYTE1(v12) = a5;
    if ( *a6 <= 0x1Cu )
    {
      return sub_B50340(12, a1, a2, a3, v12);
    }
    else
    {
      v13 = (unsigned __int8 *)sub_B50340(12, a1, a2, a3, v12);
      sub_B45260(v13, (__int64)a6, 1);
      return (__int64)v13;
    }
  }
}

// Function: sub_F7DC30
// Address: 0xf7dc30
//
__int64 __fastcall sub_F7DC30(__int64 a1, __int64 a2, unsigned __int8 *a3, unsigned __int8 *a4)
{
  int v5; // esi
  unsigned int v7; // r8d
  int v8; // edx

  v5 = *a3;
  if ( (unsigned int)(v5 - 42) > 0x11 || a1 != *((_QWORD *)a3 - 8) && a1 != *((_QWORD *)a3 - 4) )
    return 0;
  v7 = 0;
  v8 = *a4;
  if ( (unsigned int)(v8 - 42) > 0x11 || a2 != *((_QWORD *)a4 - 8) && a2 != *((_QWORD *)a4 - 4) )
    return 0;
  LOBYTE(v7) = (_BYTE)v5 == (unsigned __int8)v8;
  return v7;
}

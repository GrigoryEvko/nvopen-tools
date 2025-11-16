// Function: sub_2B41310
// Address: 0x2b41310
//
__int64 __fastcall sub_2B41310(unsigned __int8 *a1)
{
  int v1; // eax
  char v2; // bl
  bool v3; // zf
  unsigned int v5; // eax
  unsigned int v6; // r8d
  __int64 v7; // [rsp+0h] [rbp-20h] BYREF
  __int64 v8; // [rsp+8h] [rbp-18h] BYREF

  v1 = *a1;
  v7 = 0;
  v8 = 0;
  v2 = v1;
  if ( (unsigned int)(v1 - 42) <= 0x11 )
  {
    if ( *((_QWORD *)a1 - 8) )
    {
      v3 = *((_QWORD *)a1 - 4) == 0;
      v7 = *((_QWORD *)a1 - 8);
      if ( !v3 )
        return 1;
    }
  }
  v5 = sub_2B40F30((__int64)a1, &v7, &v8);
  v6 = v5;
  LOBYTE(v5) = v2 == 86;
  return v6 | v5;
}

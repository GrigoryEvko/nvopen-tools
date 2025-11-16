// Function: sub_745A90
// Address: 0x745a90
//
__int64 __fastcall sub_745A90(__int64 a1, __int64 a2, unsigned int a3, _DWORD *a4)
{
  unsigned int v6; // r15d
  __int64 v8; // rax

  *a4 = 0;
  if ( a1 == a2 )
    return 1;
  v6 = sub_745900(a1, a2);
  if ( v6 )
  {
    return 1;
  }
  else if ( a3 )
  {
    if ( (unsigned int)sub_8D3410(a1) )
    {
      v8 = sub_8D4050(a1);
      if ( (unsigned int)sub_745900(v8, a2) )
      {
        *a4 = 1;
        return a3;
      }
    }
  }
  return v6;
}

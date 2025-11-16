// Function: sub_2BE0030
// Address: 0x2be0030
//
__int64 __fastcall sub_2BE0030(__int64 a1)
{
  unsigned __int8 *v1; // r12
  int v2; // eax

  v1 = (unsigned __int8 *)(a1 + 8);
  sub_2240AE0((unsigned __int64 *)(a1 + 272), (unsigned __int64 *)(a1 + 208));
  if ( *(_QWORD *)(a1 + 184) == *(_QWORD *)(a1 + 192) )
  {
    *(_DWORD *)(a1 + 152) = 27;
    return 1;
  }
  else
  {
    v2 = *(_DWORD *)(a1 + 144);
    if ( v2 )
    {
      if ( v2 == 2 )
      {
        sub_2BDFB70((__int64)v1);
        return 1;
      }
      else
      {
        if ( v2 == 1 )
          sub_2BDF830((__int64)v1);
        return 1;
      }
    }
    else
    {
      sub_2BDF460(v1);
      return 1;
    }
  }
}

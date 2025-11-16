// Function: sub_380A2B0
// Address: 0x380a2b0
//
unsigned __int64 __fastcall sub_380A2B0(__int64 *a1, unsigned __int64 a2)
{
  int v2; // edx
  __int64 v3; // rax
  int v4; // edx
  __int16 v5; // ax

  v2 = *(_DWORD *)(a2 + 24);
  if ( v2 > 239 )
  {
    v3 = (unsigned int)(v2 - 242) < 2 ? 0x28 : 0;
  }
  else
  {
    v3 = 40;
    if ( v2 <= 237 )
      v3 = (unsigned int)(v2 - 101) < 0x30 ? 0x28 : 0;
  }
  v4 = 307;
  v5 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + v3) + 48LL)
                + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + v3 + 8));
  if ( v5 != 12 )
  {
    v4 = 308;
    if ( v5 != 13 )
    {
      v4 = 309;
      if ( v5 != 14 )
      {
        v4 = 310;
        if ( v5 != 15 )
        {
          v4 = 729;
          if ( v5 == 16 )
            v4 = 311;
        }
      }
    }
  }
  return sub_3809E10(a1, a2, v4);
}

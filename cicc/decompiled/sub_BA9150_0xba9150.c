// Function: sub_BA9150
// Address: 0xba9150
//
__int64 __fastcall sub_BA9150(__int64 a1, _DWORD *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // r13
  unsigned int v4; // r14d
  __int64 v5; // rax

  v2 = 0;
  if ( a1 )
  {
    if ( *(_BYTE *)a1 == 1 )
    {
      v3 = *(_QWORD *)(a1 + 136);
      if ( *(_BYTE *)v3 == 17 )
      {
        v4 = *(_DWORD *)(v3 + 32);
        if ( v4 > 0x40 )
        {
          if ( v4 - (unsigned int)sub_C444A0(v3 + 24) > 0x40 )
            return v2;
          v5 = **(_QWORD **)(v3 + 24);
        }
        else
        {
          v5 = *(_QWORD *)(v3 + 24);
        }
        v2 = 0;
        if ( (unsigned __int64)(v5 - 1) <= 7 )
        {
          *a2 = v5;
          return 1;
        }
      }
    }
  }
  return v2;
}

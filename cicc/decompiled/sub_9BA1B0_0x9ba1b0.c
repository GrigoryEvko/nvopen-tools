// Function: sub_9BA1B0
// Address: 0x9ba1b0
//
__int64 *__fastcall sub_9BA1B0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // ebx
  unsigned int i; // r12d
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax

  v2 = *(_DWORD *)(*(_QWORD *)(a2 + 8) + 32LL);
  *((_DWORD *)a1 + 2) = v2;
  if ( v2 > 0x40 )
  {
    sub_C43690(a1, -1, 1);
    if ( *(_BYTE *)a2 == 11 )
      goto LABEL_4;
  }
  else if ( v2 )
  {
    *a1 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v2;
    if ( *(_BYTE *)a2 == 11 )
    {
LABEL_4:
      for ( i = 0; v2 > i; ++i )
      {
        while ( 1 )
        {
          v4 = sub_AD69F0(a2, i);
          if ( (unsigned __int8)sub_AC30F0(v4) )
            break;
LABEL_5:
          if ( v2 <= ++i )
            return a1;
        }
        v5 = *a1;
        v6 = ~(1LL << i);
        if ( *((_DWORD *)a1 + 2) > 0x40u )
        {
          *(_QWORD *)(v5 + 8LL * (i >> 6)) &= v6;
          goto LABEL_5;
        }
        *a1 = v5 & v6;
      }
    }
  }
  else
  {
    *a1 = 0;
  }
  return a1;
}

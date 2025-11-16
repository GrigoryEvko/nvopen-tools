// Function: sub_9BA060
// Address: 0x9ba060
//
__int64 __fastcall sub_9BA060(unsigned __int8 *a1)
{
  __int64 v1; // rax
  int v2; // r14d
  unsigned int v3; // r13d
  __int64 v4; // rax
  unsigned __int8 *v5; // r12

  if ( *a1 <= 0x15u )
  {
    if ( !(unsigned __int8)sub_AD7930(a1) && (unsigned int)*a1 - 12 > 1 )
    {
      v1 = *((_QWORD *)a1 + 1);
      if ( *(_BYTE *)(v1 + 8) == 18 )
        return 0;
      v2 = *(_DWORD *)(v1 + 32);
      if ( v2 )
      {
        v3 = 0;
        while ( 1 )
        {
          v4 = sub_AD69F0(a1, v3);
          v5 = (unsigned __int8 *)v4;
          if ( !v4 || !(unsigned __int8)sub_AD7930(v4) && (unsigned int)*v5 - 12 > 1 )
            break;
          if ( ++v3 == v2 )
            return 1;
        }
        return 0;
      }
    }
    return 1;
  }
  return 0;
}

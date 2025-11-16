// Function: sub_8D5D50
// Address: 0x8d5d50
//
__int64 __fastcall sub_8D5D50(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 **v3; // r12
  __int64 *v4; // rdx
  __int64 *v5; // rax
  __int64 v6; // rdi

  result = 1;
  if ( a2 != a1 )
  {
    v3 = *(__int64 ***)(a1 + 112);
    if ( v3 )
    {
      while ( 1 )
      {
        v4 = v3[1];
        if ( v4 )
          break;
LABEL_9:
        if ( ((_BYTE)v3[3] & 1) == 0 && (v6 = v4[2], (*(_BYTE *)(v6 + 96) & 2) != 0) )
        {
          if ( (unsigned int)sub_8D5D50(v6, a2) )
            return 1;
          v3 = (__int64 **)*v3;
          if ( !v3 )
            return 0;
        }
        else
        {
          v3 = (__int64 **)*v3;
          if ( !v3 )
            return 0;
        }
      }
      v5 = v3[1];
      while ( v5[2] != a2 )
      {
        if ( (*(_BYTE *)(a2 + 96) & 3) == 0 )
        {
          v5 = (__int64 *)*v5;
          if ( v5 )
            continue;
        }
        goto LABEL_9;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  return result;
}

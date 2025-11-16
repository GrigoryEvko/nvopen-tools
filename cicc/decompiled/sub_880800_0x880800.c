// Function: sub_880800
// Address: 0x880800
//
__int64 __fastcall sub_880800(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 **v3; // rbx
  __int64 v4; // r12
  __int64 *v5; // rax
  __int64 v6; // rdx

  v2 = sub_8807C0(a1);
  v3 = *(__int64 ***)(a2 + 184);
  if ( !v3 )
    return 0;
  v4 = v2;
  while ( 1 )
  {
    if ( ((_BYTE)v3[5] & 0x20) != 0 )
    {
      v5 = v3[3];
      if ( (__int64 *)v4 == v5 )
        break;
      if ( v4 )
      {
        if ( v5 )
        {
          if ( dword_4F07588 )
          {
            v6 = *(_QWORD *)(v4 + 32);
            if ( v5[4] == v6 )
            {
              if ( v6 )
                break;
            }
          }
        }
      }
      if ( (unsigned int)sub_880800(a1, v5[16]) )
        break;
    }
    v3 = (__int64 **)*v3;
    if ( !v3 )
      return 0;
  }
  return 1;
}

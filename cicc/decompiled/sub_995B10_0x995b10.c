// Function: sub_995B10
// Address: 0x995b10
//
__int64 __fastcall sub_995B10(_QWORD **a1, __int64 a2)
{
  unsigned int v2; // r13d
  bool v3; // al
  int v5; // r13d
  char v6; // r14
  unsigned int v7; // r15d
  __int64 v8; // rax
  unsigned int v9; // r14d
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int v12; // r13d

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( !v2 )
      goto LABEL_19;
    if ( v2 <= 0x40 )
      v3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v2) == *(_QWORD *)(a2 + 24);
    else
      v3 = v2 == (unsigned int)sub_C445E0(a2 + 24);
    goto LABEL_5;
  }
  v10 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v11 = sub_AD7630(a2, 0);
  if ( !v11 || *(_BYTE *)v11 != 17 )
  {
    if ( *(_BYTE *)(v10 + 8) == 17 )
    {
      v5 = *(_DWORD *)(v10 + 32);
      if ( v5 )
      {
        v6 = 0;
        v7 = 0;
        while ( 1 )
        {
          v8 = sub_AD69F0(a2, v7);
          if ( !v8 )
            break;
          if ( *(_BYTE *)v8 != 13 )
          {
            if ( *(_BYTE *)v8 != 17 )
              return 0;
            v9 = *(_DWORD *)(v8 + 32);
            if ( v9 )
            {
              if ( v9 <= 0x40 )
              {
                if ( *(_QWORD *)(v8 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) )
                  return 0;
              }
              else if ( v9 != (unsigned int)sub_C445E0(v8 + 24) )
              {
                return 0;
              }
            }
            v6 = 1;
          }
          if ( v5 == ++v7 )
          {
            if ( !v6 )
              return 0;
            goto LABEL_19;
          }
        }
      }
    }
    return 0;
  }
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 )
  {
    if ( v12 <= 0x40 )
      v3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == *(_QWORD *)(v11 + 24);
    else
      v3 = v12 == (unsigned int)sub_C445E0(v11 + 24);
LABEL_5:
    if ( !v3 )
      return 0;
  }
LABEL_19:
  if ( *a1 )
    **a1 = a2;
  return 1;
}

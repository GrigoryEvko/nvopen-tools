// Function: sub_10E4C50
// Address: 0x10e4c50
//
_BOOL8 __fastcall sub_10E4C50(__int64 a1, _BYTE *a2)
{
  _BOOL4 v2; // r12d
  unsigned __int64 v4; // rax
  _BOOL4 v5; // edx
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rdx
  bool v9; // r13
  unsigned int v10; // r13d
  bool v11; // al
  __int64 v12; // r14
  _BYTE *v13; // rax
  unsigned int v14; // ebx
  int v15; // r14d
  unsigned int v16; // r15d
  __int64 v17; // rax
  unsigned int v18; // r13d

  v2 = 0;
  if ( *a2 == 82 )
  {
    v4 = sub_B53900((__int64)a2);
    sub_B53630(v4, *(_QWORD *)a1);
    v2 = v5;
    if ( !v5 )
      return 0;
    v6 = *((_QWORD *)a2 - 8);
    if ( !v6 )
      return 0;
    **(_QWORD **)(a1 + 8) = v6;
    v7 = *((_QWORD *)a2 - 4);
    if ( *(_BYTE *)v7 > 0x15u )
      return 0;
    v9 = sub_AC30F0(v7);
    if ( v9 )
      return v2;
    if ( *(_BYTE *)v7 == 17 )
    {
      v10 = *(_DWORD *)(v7 + 32);
      if ( v10 <= 0x40 )
        v11 = *(_QWORD *)(v7 + 24) == 0;
      else
        v11 = v10 == (unsigned int)sub_C444A0(v7 + 24);
    }
    else
    {
      v12 = *(_QWORD *)(v7 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v12 + 8) - 17 > 1 )
        return 0;
      v13 = sub_AD7630(v7, 0, v8);
      if ( !v13 || *v13 != 17 )
      {
        if ( *(_BYTE *)(v12 + 8) == 17 )
        {
          v15 = *(_DWORD *)(v12 + 32);
          if ( v15 )
          {
            v16 = 0;
            while ( 1 )
            {
              v17 = sub_AD69F0((unsigned __int8 *)v7, v16);
              if ( !v17 )
                break;
              if ( *(_BYTE *)v17 != 13 )
              {
                if ( *(_BYTE *)v17 != 17 )
                  return 0;
                v18 = *(_DWORD *)(v17 + 32);
                if ( v18 <= 0x40 )
                {
                  if ( *(_QWORD *)(v17 + 24) )
                    return 0;
                }
                else if ( v18 != (unsigned int)sub_C444A0(v17 + 24) )
                {
                  return 0;
                }
                v9 = v2;
              }
              if ( v15 == ++v16 )
              {
                if ( v9 )
                  return v2;
                return 0;
              }
            }
          }
        }
        return 0;
      }
      v14 = *((_DWORD *)v13 + 8);
      if ( v14 <= 0x40 )
      {
        if ( !*((_QWORD *)v13 + 3) )
          return v2;
        return 0;
      }
      v11 = v14 == (unsigned int)sub_C444A0((__int64)(v13 + 24));
    }
    if ( v11 )
      return v2;
    return 0;
  }
  return v2;
}

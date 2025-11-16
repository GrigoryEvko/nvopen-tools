// Function: sub_10081F0
// Address: 0x10081f0
//
__int64 __fastcall sub_10081F0(__int64 **a1, __int64 a2)
{
  unsigned int v2; // r13d
  bool v3; // al
  __int64 v5; // r13
  __int64 v6; // rdx
  _BYTE *v7; // rax
  unsigned int v8; // r13d
  int v9; // r13d
  char v10; // r14
  unsigned int v11; // r15d
  __int64 v12; // rax
  unsigned int v13; // r14d

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *(_DWORD *)(a2 + 32);
    if ( v2 <= 0x40 )
      v3 = *(_QWORD *)(a2 + 24) == 0;
    else
      v3 = v2 == (unsigned int)sub_C444A0(a2 + 24);
LABEL_4:
    if ( v3 )
      goto LABEL_5;
    return 0;
  }
  v5 = *(_QWORD *)(a2 + 8);
  v6 = (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17;
  if ( (unsigned int)v6 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v7 = sub_AD7630(a2, 0, v6);
  if ( !v7 || *v7 != 17 )
  {
    if ( *(_BYTE *)(v5 + 8) == 17 )
    {
      v9 = *(_DWORD *)(v5 + 32);
      if ( v9 )
      {
        v10 = 0;
        v11 = 0;
        while ( 1 )
        {
          v12 = sub_AD69F0((unsigned __int8 *)a2, v11);
          if ( !v12 )
            break;
          if ( *(_BYTE *)v12 != 13 )
          {
            if ( *(_BYTE *)v12 != 17 )
              return 0;
            v13 = *(_DWORD *)(v12 + 32);
            if ( v13 <= 0x40 )
            {
              if ( *(_QWORD *)(v12 + 24) )
                return 0;
            }
            else if ( v13 != (unsigned int)sub_C444A0(v12 + 24) )
            {
              return 0;
            }
            v10 = 1;
          }
          if ( v9 == ++v11 )
          {
            if ( v10 )
              goto LABEL_5;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v8 = *((_DWORD *)v7 + 8);
  if ( v8 > 0x40 )
  {
    v3 = v8 == (unsigned int)sub_C444A0((__int64)(v7 + 24));
    goto LABEL_4;
  }
  if ( *((_QWORD *)v7 + 3) )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = a2;
  return 1;
}

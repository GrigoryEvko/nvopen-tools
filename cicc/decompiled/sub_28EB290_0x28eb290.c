// Function: sub_28EB290
// Address: 0x28eb290
//
__int64 __fastcall sub_28EB290(__int64 **a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r13d
  bool v4; // al
  __int64 v6; // r13
  __int64 v7; // rdx
  _BYTE *v8; // rax
  unsigned int v9; // r13d
  int v10; // r13d
  char v11; // r14
  unsigned int v12; // r15d
  __int64 v13; // rax
  unsigned int v14; // r14d

  v2 = *(_QWORD *)(a2 - 64);
  if ( *(_BYTE *)v2 == 17 )
  {
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      v4 = *(_QWORD *)(v2 + 24) == 0;
    else
      v4 = v3 == (unsigned int)sub_C444A0(v2 + 24);
LABEL_4:
    if ( v4 )
      goto LABEL_5;
    return 0;
  }
  v6 = *(_QWORD *)(v2 + 8);
  v7 = (unsigned int)*(unsigned __int8 *)(v6 + 8) - 17;
  if ( (unsigned int)v7 > 1 || *(_BYTE *)v2 > 0x15u )
    return 0;
  v8 = sub_AD7630(v2, 0, v7);
  if ( !v8 || *v8 != 17 )
  {
    if ( *(_BYTE *)(v6 + 8) == 17 )
    {
      v10 = *(_DWORD *)(v6 + 32);
      if ( v10 )
      {
        v11 = 0;
        v12 = 0;
        while ( 1 )
        {
          v13 = sub_AD69F0((unsigned __int8 *)v2, v12);
          if ( !v13 )
            break;
          if ( *(_BYTE *)v13 != 13 )
          {
            if ( *(_BYTE *)v13 != 17 )
              return 0;
            v14 = *(_DWORD *)(v13 + 32);
            if ( v14 <= 0x40 )
            {
              if ( *(_QWORD *)(v13 + 24) )
                return 0;
            }
            else if ( v14 != (unsigned int)sub_C444A0(v13 + 24) )
            {
              return 0;
            }
            v11 = 1;
          }
          if ( v10 == ++v12 )
          {
            if ( v11 )
              goto LABEL_5;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v9 = *((_DWORD *)v8 + 8);
  if ( v9 > 0x40 )
  {
    v4 = v9 == (unsigned int)sub_C444A0((__int64)(v8 + 24));
    goto LABEL_4;
  }
  if ( *((_QWORD *)v8 + 3) )
    return 0;
LABEL_5:
  if ( *a1 )
    **a1 = v2;
  return 1;
}

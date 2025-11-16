// Function: sub_1155650
// Address: 0x1155650
//
__int64 __fastcall sub_1155650(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r13d
  bool v4; // al
  __int64 *v5; // rax
  __int64 v7; // r13
  __int64 v8; // rdx
  _BYTE *v9; // rax
  unsigned int v10; // r13d
  int v11; // r13d
  char v12; // r14
  unsigned int v13; // r15d
  __int64 v14; // rax
  unsigned int v15; // r14d

  v2 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v2 == 17 )
  {
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      v4 = *(_QWORD *)(v2 + 24) == 1;
    else
      v4 = v3 - 1 == (unsigned int)sub_C444A0(v2 + 24);
LABEL_4:
    if ( v4 )
      goto LABEL_5;
    return 0;
  }
  v7 = *(_QWORD *)(v2 + 8);
  v8 = (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17;
  if ( (unsigned int)v8 > 1 || *(_BYTE *)v2 > 0x15u )
    return 0;
  v9 = sub_AD7630(v2, 0, v8);
  if ( !v9 || *v9 != 17 )
  {
    if ( *(_BYTE *)(v7 + 8) == 17 )
    {
      v11 = *(_DWORD *)(v7 + 32);
      if ( v11 )
      {
        v12 = 0;
        v13 = 0;
        while ( 1 )
        {
          v14 = sub_AD69F0((unsigned __int8 *)v2, v13);
          if ( !v14 )
            break;
          if ( *(_BYTE *)v14 != 13 )
          {
            if ( *(_BYTE *)v14 != 17 )
              return 0;
            v15 = *(_DWORD *)(v14 + 32);
            if ( v15 <= 0x40 )
            {
              if ( *(_QWORD *)(v14 + 24) != 1 )
                return 0;
            }
            else if ( (unsigned int)sub_C444A0(v14 + 24) != v15 - 1 )
            {
              return 0;
            }
            v12 = 1;
          }
          if ( v11 == ++v13 )
          {
            if ( v12 )
              goto LABEL_5;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v10 = *((_DWORD *)v9 + 8);
  if ( v10 > 0x40 )
  {
    v4 = v10 - 1 == (unsigned int)sub_C444A0((__int64)(v9 + 24));
    goto LABEL_4;
  }
  if ( *((_QWORD *)v9 + 3) != 1 )
    return 0;
LABEL_5:
  v5 = *(__int64 **)(a1 + 8);
  if ( v5 )
    *v5 = v2;
  return 1;
}

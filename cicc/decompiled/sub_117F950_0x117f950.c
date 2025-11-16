// Function: sub_117F950
// Address: 0x117f950
//
__int64 __fastcall sub_117F950(__int64 **a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  _BYTE *v6; // rax
  int v7; // r13d
  char v8; // r14
  unsigned int v9; // r15d
  char *v10; // rax
  char v11; // al

  if ( *(_BYTE *)a2 == 17 )
  {
    v2 = *a1;
    if ( !*a1 )
    {
LABEL_5:
      *a1[1] = a2;
      return 1;
    }
LABEL_3:
    *v2 = a2;
    goto LABEL_4;
  }
  v4 = *(_QWORD *)(a2 + 8);
  v5 = (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17;
  if ( (unsigned int)v5 > 1 || *(_BYTE *)a2 > 0x15u )
    return 0;
  v6 = sub_AD7630(a2, 0, v5);
  if ( !v6 || *v6 != 17 )
  {
    if ( *(_BYTE *)(v4 + 8) == 17 )
    {
      v7 = *(_DWORD *)(v4 + 32);
      if ( v7 )
      {
        v8 = 0;
        v9 = 0;
        while ( 1 )
        {
          v10 = (char *)sub_AD69F0((unsigned __int8 *)a2, v9);
          if ( !v10 )
            break;
          v11 = *v10;
          if ( v11 != 13 )
          {
            if ( v11 != 17 )
              return 0;
            v8 = 1;
          }
          if ( v7 == ++v9 )
          {
            if ( v8 )
              goto LABEL_11;
            return 0;
          }
        }
      }
    }
    return 0;
  }
LABEL_11:
  v2 = *a1;
  if ( *a1 )
    goto LABEL_3;
LABEL_4:
  if ( *(_BYTE *)a2 <= 0x15u )
    goto LABEL_5;
  return 0;
}

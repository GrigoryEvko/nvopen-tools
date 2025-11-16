// Function: sub_10B14D0
// Address: 0x10b14d0
//
bool __fastcall sub_10B14D0(__int64 **a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rdx
  _BYTE *v10; // rax
  unsigned int v11; // r14d
  char v12; // r15
  unsigned int v13; // r14d
  __int64 v14; // rax
  unsigned int v15; // r15d
  int v16; // [rsp-3Ch] [rbp-3Ch]

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 17 )
  {
    v6 = *(_DWORD *)(v5 + 32);
    if ( v6 <= 0x40 )
      v7 = *(_QWORD *)(v5 + 24) == 0;
    else
      v7 = v6 == (unsigned int)sub_C444A0(v5 + 24);
  }
  else
  {
    v8 = *(_QWORD *)(v5 + 8);
    v9 = (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17;
    if ( (unsigned int)v9 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v10 = sub_AD7630(v5, 0, v9);
    if ( !v10 || *v10 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v16 = *(_DWORD *)(v8 + 32);
        if ( v16 )
        {
          v12 = 0;
          v13 = 0;
          while ( 1 )
          {
            v14 = sub_AD69F0((unsigned __int8 *)v5, v13);
            if ( !v14 )
              break;
            if ( *(_BYTE *)v14 != 13 )
            {
              if ( *(_BYTE *)v14 != 17 )
                return 0;
              v15 = *(_DWORD *)(v14 + 32);
              if ( v15 <= 0x40 )
              {
                if ( *(_QWORD *)(v14 + 24) )
                  return 0;
              }
              else if ( v15 != (unsigned int)sub_C444A0(v14 + 24) )
              {
                return 0;
              }
              v12 = 1;
            }
            if ( v16 == ++v13 )
            {
              if ( v12 )
                goto LABEL_7;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v11 = *((_DWORD *)v10 + 8);
    if ( v11 <= 0x40 )
    {
      if ( !*((_QWORD *)v10 + 3) )
        goto LABEL_7;
      return 0;
    }
    v7 = v11 == (unsigned int)sub_C444A0((__int64)(v10 + 24));
  }
  if ( !v7 )
    return 0;
LABEL_7:
  if ( *a1 )
    **a1 = v5;
  return *a1[1] == *((_QWORD *)a3 - 4);
}

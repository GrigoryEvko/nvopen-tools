// Function: sub_99C280
// Address: 0x99c280
//
bool __fastcall sub_99C280(__int64 a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // r14
  __int64 v9; // rax
  unsigned int v10; // r14d
  char v11; // r15
  unsigned int v12; // r14d
  __int64 v13; // rax
  unsigned int v14; // r15d
  int v15; // [rsp-3Ch] [rbp-3Ch]

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
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 > 1 || *(_BYTE *)v5 > 0x15u )
      return 0;
    v9 = sub_AD7630(v5, 0);
    if ( !v9 || *(_BYTE *)v9 != 17 )
    {
      if ( *(_BYTE *)(v8 + 8) == 17 )
      {
        v15 = *(_DWORD *)(v8 + 32);
        if ( v15 )
        {
          v11 = 0;
          v12 = 0;
          while ( 1 )
          {
            v13 = sub_AD69F0(v5, v12);
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
            if ( v15 == ++v12 )
            {
              if ( v11 )
                goto LABEL_7;
              return 0;
            }
          }
        }
      }
      return 0;
    }
    v10 = *(_DWORD *)(v9 + 32);
    if ( v10 <= 0x40 )
    {
      if ( !*(_QWORD *)(v9 + 24) )
        goto LABEL_7;
      return 0;
    }
    v7 = v10 == (unsigned int)sub_C444A0(v9 + 24);
  }
  if ( !v7 )
    return 0;
LABEL_7:
  if ( *(_QWORD *)a1 )
    **(_QWORD **)a1 = v5;
  return *((_QWORD *)a3 - 4) == *(_QWORD *)(a1 + 8);
}

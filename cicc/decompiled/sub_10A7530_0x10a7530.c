// Function: sub_10A7530
// Address: 0x10a7530
//
__int64 __fastcall sub_10A7530(__int64 **a1, int a2, unsigned __int8 *a3)
{
  __int64 v5; // r13
  unsigned int v6; // r14d
  bool v7; // al
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rdx
  _BYTE *v11; // rax
  unsigned int v12; // r14d
  char v13; // r15
  unsigned int v14; // r14d
  __int64 v15; // rax
  unsigned int v16; // r15d
  int v17; // [rsp-3Ch] [rbp-3Ch]

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
LABEL_6:
    if ( !v7 )
      return 0;
    goto LABEL_7;
  }
  v9 = *(_QWORD *)(v5 + 8);
  v10 = (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17;
  if ( (unsigned int)v10 > 1 || *(_BYTE *)v5 > 0x15u )
    return 0;
  v11 = sub_AD7630(v5, 0, v10);
  if ( !v11 || *v11 != 17 )
  {
    if ( *(_BYTE *)(v9 + 8) == 17 )
    {
      v17 = *(_DWORD *)(v9 + 32);
      if ( v17 )
      {
        v13 = 0;
        v14 = 0;
        while ( 1 )
        {
          v15 = sub_AD69F0((unsigned __int8 *)v5, v14);
          if ( !v15 )
            break;
          if ( *(_BYTE *)v15 != 13 )
          {
            if ( *(_BYTE *)v15 != 17 )
              return 0;
            v16 = *(_DWORD *)(v15 + 32);
            if ( v16 <= 0x40 )
            {
              if ( *(_QWORD *)(v15 + 24) )
                return 0;
            }
            else if ( v16 != (unsigned int)sub_C444A0(v15 + 24) )
            {
              return 0;
            }
            v13 = 1;
          }
          if ( v17 == ++v14 )
          {
            if ( v13 )
              goto LABEL_7;
            return 0;
          }
        }
      }
    }
    return 0;
  }
  v12 = *((_DWORD *)v11 + 8);
  if ( v12 > 0x40 )
  {
    v7 = v12 == (unsigned int)sub_C444A0((__int64)(v11 + 24));
    goto LABEL_6;
  }
  if ( *((_QWORD *)v11 + 3) )
    return 0;
LABEL_7:
  if ( *a1 )
    **a1 = v5;
  v8 = *((_QWORD *)a3 - 4);
  if ( v8 )
  {
    *a1[1] = v8;
    return 1;
  }
  return 0;
}

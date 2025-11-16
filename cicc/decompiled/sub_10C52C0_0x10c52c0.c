// Function: sub_10C52C0
// Address: 0x10c52c0
//
__int64 __fastcall sub_10C52C0(__int64 a1, char *a2)
{
  __int64 v2; // rax
  char v5; // al
  __int64 v6; // rax
  __int64 v7; // rdi
  _BYTE *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  _BYTE *v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rax

  v2 = *((_QWORD *)a2 + 2);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  v5 = *a2;
  if ( *a2 != 67 )
  {
LABEL_5:
    if ( v5 == 56 )
    {
      v6 = *((_QWORD *)a2 - 8);
      if ( v6 )
      {
        **(_QWORD **)(a1 + 24) = v6;
        v7 = *((_QWORD *)a2 - 4);
        if ( *(_BYTE *)v7 == 17 )
        {
          **(_QWORD **)(a1 + 32) = v7 + 24;
          return 1;
        }
        v11 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
        if ( (unsigned int)v11 <= 1 && *(_BYTE *)v7 <= 0x15u )
        {
          v12 = sub_AD7630(v7, *(unsigned __int8 *)(a1 + 40), v11);
          if ( v12 )
          {
            if ( *v12 == 17 )
            {
              **(_QWORD **)(a1 + 32) = v12 + 24;
              return 1;
            }
          }
        }
      }
    }
    return 0;
  }
  v8 = (_BYTE *)*((_QWORD *)a2 - 4);
  if ( *v8 == 56 )
  {
    v9 = *((_QWORD *)v8 - 8);
    if ( v9 )
    {
      **(_QWORD **)a1 = v9;
      v10 = *((_QWORD *)v8 - 4);
      if ( *(_BYTE *)v10 == 17 )
      {
        **(_QWORD **)(a1 + 8) = v10 + 24;
        return 1;
      }
      v13 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
      if ( (unsigned int)v13 <= 1 && *(_BYTE *)v10 <= 0x15u )
      {
        v14 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 16), v13);
        if ( v14 )
        {
          if ( *v14 == 17 )
          {
            **(_QWORD **)(a1 + 8) = v14 + 24;
            return 1;
          }
        }
      }
      v5 = *a2;
      goto LABEL_5;
    }
  }
  return 0;
}

// Function: sub_10C3E60
// Address: 0x10c3e60
//
__int64 __fastcall sub_10C3E60(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rdx

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a2 - 64);
  v4 = *(_QWORD *)(v3 + 16);
  if ( !v4 || *(_QWORD *)(v4 + 8) || *(_BYTE *)v3 != 68 || (v8 = *(_QWORD *)(v3 - 32)) == 0 )
  {
LABEL_3:
    v5 = *(_QWORD *)(a2 - 32);
    goto LABEL_4;
  }
  **(_QWORD **)a1 = v8;
  v5 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v5 == 17 )
  {
LABEL_11:
    **(_QWORD **)(a1 + 8) = v5 + 24;
    return 1;
  }
  v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
  if ( (unsigned int)v12 <= 1 && *(_BYTE *)v5 <= 0x15u )
  {
    v11 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 16), v12);
    if ( v11 && *v11 == 17 )
    {
LABEL_21:
      **(_QWORD **)(a1 + 8) = v11 + 24;
      return 1;
    }
    goto LABEL_3;
  }
LABEL_4:
  v6 = *(_QWORD *)(v5 + 16);
  if ( v6 )
  {
    if ( !*(_QWORD *)(v6 + 8) && *(_BYTE *)v5 == 68 )
    {
      v9 = *(_QWORD *)(v5 - 32);
      if ( v9 )
      {
        **(_QWORD **)a1 = v9;
        v5 = *(_QWORD *)(a2 - 64);
        if ( *(_BYTE *)v5 == 17 )
          goto LABEL_11;
        v10 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
        if ( (unsigned int)v10 <= 1 && *(_BYTE *)v5 <= 0x15u )
        {
          v11 = sub_AD7630(v5, *(unsigned __int8 *)(a1 + 16), v10);
          if ( v11 )
          {
            if ( *v11 == 17 )
              goto LABEL_21;
          }
        }
      }
    }
  }
  return 0;
}

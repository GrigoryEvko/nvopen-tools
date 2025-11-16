// Function: sub_10DC410
// Address: 0x10dc410
//
__int64 __fastcall sub_10DC410(__int64 a1, int a2, unsigned __int8 *a3)
{
  unsigned int v3; // r12d
  unsigned int v7; // eax
  _BYTE *v8; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  _BYTE *v12; // rax
  _BYTE *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v7 = sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 8));
  v8 = (_BYTE *)*((_QWORD *)a3 - 4);
  v3 = v7;
  if ( (_BYTE)v7 )
  {
    if ( *v8 == 56 )
    {
      v9 = *((_QWORD *)v8 - 8);
      if ( v9 )
      {
        **(_QWORD **)(a1 + 8) = v9;
        v10 = *((_QWORD *)v8 - 4);
        if ( *(_BYTE *)v10 == 17 )
        {
LABEL_8:
          **(_QWORD **)(a1 + 16) = v10 + 24;
          return v3;
        }
        v11 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
        if ( (unsigned int)v11 <= 1 && *(_BYTE *)v10 <= 0x15u )
        {
          v12 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 24), v11);
          if ( v12 )
          {
            if ( *v12 == 17 )
            {
LABEL_22:
              **(_QWORD **)(a1 + 16) = v12 + 24;
              return v3;
            }
          }
        }
        v8 = (_BYTE *)*((_QWORD *)a3 - 4);
      }
    }
  }
  v3 = sub_995B10((_QWORD **)a1, (__int64)v8);
  if ( (_BYTE)v3 )
  {
    v13 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v13 == 56 )
    {
      v14 = *((_QWORD *)v13 - 8);
      if ( v14 )
      {
        **(_QWORD **)(a1 + 8) = v14;
        v10 = *((_QWORD *)v13 - 4);
        if ( *(_BYTE *)v10 == 17 )
          goto LABEL_8;
        v15 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v10 + 8) + 8LL) - 17;
        if ( (unsigned int)v15 <= 1 && *(_BYTE *)v10 <= 0x15u )
        {
          v12 = sub_AD7630(v10, *(unsigned __int8 *)(a1 + 24), v15);
          if ( v12 )
          {
            if ( *v12 == 17 )
              goto LABEL_22;
          }
        }
      }
    }
  }
  return 0;
}

// Function: sub_1158710
// Address: 0x1158710
//
__int64 __fastcall sub_1158710(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rax
  _BYTE *v4; // rax
  __int64 v6; // rdx
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rax

  if ( a2 )
  {
    v3 = *(_BYTE **)(a2 - 64);
    if ( *v3 == 55 )
    {
      v6 = *((_QWORD *)v3 - 8);
      if ( v6 )
      {
        **(_QWORD **)a1 = v6;
        v7 = *((_QWORD *)v3 - 4);
        if ( *(_BYTE *)v7 == 17 )
        {
          **(_QWORD **)(a1 + 8) = v7 + 24;
          goto LABEL_9;
        }
        v10 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v7 + 8) + 8LL) - 17;
        if ( (unsigned int)v10 <= 1 && *(_BYTE *)v7 <= 0x15u )
        {
          v11 = sub_AD7630(v7, *(unsigned __int8 *)(a1 + 16), v10);
          if ( v11 )
          {
            if ( *v11 == 17 )
            {
              **(_QWORD **)(a1 + 8) = v11 + 24;
LABEL_9:
              v4 = *(_BYTE **)(a2 - 32);
              if ( v4 )
              {
LABEL_15:
                **(_QWORD **)(a1 + 24) = v4;
                return 1;
              }
LABEL_4:
              if ( *v4 != 55 )
                return 0;
              v8 = *((_QWORD *)v4 - 8);
              if ( !v8 )
                return 0;
              **(_QWORD **)a1 = v8;
              v9 = *((_QWORD *)v4 - 4);
              if ( *(_BYTE *)v9 == 17 )
              {
                **(_QWORD **)(a1 + 8) = v9 + 24;
              }
              else
              {
                v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v9 + 8) + 8LL) - 17;
                if ( (unsigned int)v12 > 1 )
                  return 0;
                if ( *(_BYTE *)v9 > 0x15u )
                  return 0;
                v13 = sub_AD7630(v9, *(unsigned __int8 *)(a1 + 16), v12);
                if ( !v13 || *v13 != 17 )
                  return 0;
                **(_QWORD **)(a1 + 8) = v13 + 24;
              }
              v4 = *(_BYTE **)(a2 - 64);
              if ( v4 )
                goto LABEL_15;
              return 0;
            }
          }
        }
      }
    }
    v4 = *(_BYTE **)(a2 - 32);
    goto LABEL_4;
  }
  return 0;
}

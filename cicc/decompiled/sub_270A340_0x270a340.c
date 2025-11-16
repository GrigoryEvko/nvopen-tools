// Function: sub_270A340
// Address: 0x270a340
//
__int64 __fastcall sub_270A340(__int64 a1, int a2)
{
  __int64 v2; // r12
  unsigned int v3; // r14d
  __int64 v5; // r13
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v8; // rdx
  __int64 v9; // [rsp+0h] [rbp-40h]

  v2 = *(_QWORD *)(a1 - 32);
  if ( v2 )
  {
    if ( !*(_BYTE *)v2 && *(_QWORD *)(a1 + 80) == *(_QWORD *)(v2 + 24) && !sub_B2FC80(*(_QWORD *)(a1 - 32)) )
    {
      v3 = sub_B2FC00((_BYTE *)v2);
      if ( !(_BYTE)v3 )
      {
        v5 = *(_QWORD *)(v2 + 80);
        v9 = v2 + 72;
        if ( v2 + 72 == v5 )
          return v3;
        while ( 1 )
        {
          if ( !v5 )
            BUG();
          v6 = *(_QWORD *)(v5 + 32);
          v7 = v5 + 24;
          if ( v5 + 24 != v6 )
            break;
LABEL_20:
          v5 = *(_QWORD *)(v5 + 8);
          if ( v9 == v5 )
            return v3;
        }
        while ( 1 )
        {
          while ( 1 )
          {
            if ( !v6 )
              BUG();
            if ( (unsigned __int8)(*(_BYTE *)(v6 - 24) - 34) <= 0x33u )
            {
              v8 = 0x8000000000041LL;
              if ( _bittest64(&v8, (unsigned int)*(unsigned __int8 *)(v6 - 24) - 34) )
              {
                if ( a2 != 3 && !(unsigned __int8)sub_B49E20(v6 - 24) )
                  break;
              }
            }
            v6 = *(_QWORD *)(v6 + 8);
            if ( v7 == v6 )
              goto LABEL_20;
          }
          if ( (unsigned __int8)sub_270A340(v6 - 24, (unsigned int)(a2 + 1)) )
            break;
          v6 = *(_QWORD *)(v6 + 8);
          if ( v7 == v6 )
            goto LABEL_20;
        }
      }
    }
  }
  return 1;
}

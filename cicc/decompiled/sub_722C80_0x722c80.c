// Function: sub_722C80
// Address: 0x722c80
//
int __fastcall sub_722C80(__int64 *a1, int a2)
{
  _QWORD *v2; // rax
  __int64 i; // r13
  __int64 v4; // r12
  __int64 v5; // r14
  __int64 v6; // rbx

  v2 = &qword_4F076A8;
  for ( i = qword_4F076A8; i; i = *(_QWORD *)(i + 16) )
  {
    if ( a2 )
    {
      if ( !*(_DWORD *)(i + 8) )
        continue;
      v2 = &qword_4F076A8;
      v4 = qword_4F076A8;
    }
    else
    {
      v4 = *(_QWORD *)(i + 16);
    }
    if ( v4 )
    {
      v5 = 0;
      do
      {
        v6 = v4;
        v4 = *(_QWORD *)(v4 + 16);
        if ( a2 )
        {
          if ( i != v6 )
          {
            LODWORD(v2) = *(_DWORD *)(v6 + 8);
            if ( !(_DWORD)v2 )
            {
LABEL_18:
              LODWORD(v2) = sub_722B80(*(unsigned __int8 **)i, *(unsigned __int8 **)v6, 0);
              if ( !(_DWORD)v2 )
              {
                if ( v5 )
                  *(_QWORD *)(v5 + 16) = *(_QWORD *)(v6 + 16);
                if ( qword_4F076A8 == v6 )
                  qword_4F076A8 = *(_QWORD *)(v6 + 16);
                if ( *(_QWORD *)(i + 16) == v6 )
                  *(_QWORD *)(i + 16) = *(_QWORD *)(v6 + 16);
                if ( *a1 == v6 )
                  *a1 = v5;
                if ( a2 )
                  sub_684860(0x71Bu, *(_QWORD *)v6);
                v2 = (_QWORD *)qword_4F07940;
                qword_4F07940 = v6;
                *(_QWORD *)(v6 + 16) = v2;
                continue;
              }
            }
            v5 = v6;
            continue;
          }
        }
        else
        {
          if ( !*(_DWORD *)(i + 8) )
          {
            v2 = (_QWORD *)*a1;
            if ( *a1 )
            {
              if ( v2[2] == v6 )
                break;
            }
          }
          if ( i != v6 )
            goto LABEL_18;
        }
        v5 = i;
      }
      while ( v4 );
    }
  }
  return (int)v2;
}

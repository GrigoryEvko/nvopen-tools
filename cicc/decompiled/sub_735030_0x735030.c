// Function: sub_735030
// Address: 0x735030
//
__int64 sub_735030()
{
  _QWORD *v0; // r12
  _QWORD *i; // r13
  _QWORD *v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __int64 v6; // rdx
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 j; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // r14
  __int64 *v14; // r15
  __int64 v15; // rax
  __int64 result; // rax

  v0 = (_QWORD *)unk_4F072C0;
  if ( unk_4F072C0 )
  {
    for ( i = 0; ; i = v2 )
    {
      v2 = v0;
      v0 = (_QWORD *)*v0;
      if ( !*(_DWORD *)(v2[1] + 160LL) )
      {
        v3 = v2[4];
        if ( v3 )
        {
          v4 = 0;
          do
          {
            while ( 1 )
            {
              v5 = v3;
              v3 = *(_QWORD *)(v3 + 112);
              if ( *(char *)(v5 - 8) >= 0 )
                break;
              v4 = v5;
              if ( !v3 )
                goto LABEL_13;
            }
            if ( v4 )
              *(_QWORD *)(v4 + 112) = v3;
            else
              v2[4] = v3;
            *(_QWORD *)(v5 + 112) = 0;
          }
          while ( v3 );
        }
LABEL_13:
        v6 = v2[3];
        if ( v6 )
        {
          v7 = 0;
          do
          {
            v8 = v6;
            v6 = *(_QWORD *)(v6 + 112);
            for ( j = v8; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            {
              if ( *(_QWORD *)(j + 8) )
                break;
            }
            if ( *(char *)(j - 8) < 0 )
            {
              v7 = v8;
            }
            else
            {
              if ( v7 )
                *(_QWORD *)(v7 + 112) = v6;
              else
                v2[3] = v6;
              *(_QWORD *)(v8 + 112) = 0;
            }
          }
          while ( v6 );
        }
        v10 = v2[5];
        if ( v10 )
        {
          v11 = 0;
          do
          {
            while ( 1 )
            {
              v12 = v10;
              v10 = *(_QWORD *)(v10 + 112);
              if ( *(char *)(v12 - 8) >= 0 )
                break;
              v11 = v12;
              if ( !v10 )
                goto LABEL_31;
            }
            if ( v11 )
              *(_QWORD *)(v11 + 112) = v10;
            else
              v2[5] = v10;
            *(_QWORD *)(v12 + 112) = 0;
          }
          while ( v10 );
        }
LABEL_31:
        v13 = (_QWORD *)v2[6];
        if ( v13 )
        {
          do
          {
            while ( 1 )
            {
              v14 = (__int64 *)v13[1];
              if ( v14 )
                break;
              v13 = (_QWORD *)*v13;
              if ( !v13 )
                goto LABEL_48;
            }
            do
            {
              v15 = sub_72A270(v14[3], *((_BYTE *)v14 + 16));
              if ( v15 )
                *(_QWORD *)(v15 + 96) = 0;
              v14 = (__int64 *)*v14;
            }
            while ( v14 );
            v13 = (_QWORD *)*v13;
          }
          while ( v13 );
LABEL_48:
          v2[6] = 0;
        }
        if ( !v2[4] && !v2[3] && !v2[5] )
        {
          if ( i )
            *i = v0;
          else
            unk_4F072C0 = v0;
          *v2 = 0;
          v2 = i;
        }
      }
      if ( !v0 )
        break;
    }
  }
  else
  {
    v2 = 0;
  }
  result = unk_4D03FF0;
  *(_QWORD *)(unk_4D03FF0 + 352LL) = v2;
  return result;
}

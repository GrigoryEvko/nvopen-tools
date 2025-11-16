// Function: sub_819BD0
// Address: 0x819bd0
//
__int64 __fastcall sub_819BD0(_QWORD *a1)
{
  __int64 v1; // r13
  bool v2; // zf
  char *v3; // rcx
  char v4; // si
  char *v5; // r12
  char *v6; // r15
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // r14
  char *v9; // rdi
  __int64 v10; // rcx
  unsigned __int64 j; // rax
  char v13; // di
  unsigned __int64 k; // rax
  unsigned __int64 v15; // rax
  __int64 i; // rbx

  v1 = a1[11];
  v2 = *(_QWORD *)(v1 + 16) == 0;
  qword_4F06C40 = 0;
  if ( !v2 )
  {
    sub_7295A0("#define ");
    sub_7295A0(*(char **)(*a1 + 8LL));
    if ( (*(_BYTE *)v1 & 1) == 0 )
    {
      sub_729660(40);
      for ( i = *(_QWORD *)(v1 + 8); i; i = *(_QWORD *)(i + 8) )
      {
        if ( (*(_BYTE *)v1 & 8) != 0 && !*(_QWORD *)(i + 8) )
        {
          if ( unk_4D04780 )
            sub_7295A0(*(char **)i);
          sub_7295A0("...");
        }
        else
        {
          sub_7295A0(*(char **)i);
        }
        if ( !*(_QWORD *)(i + 8) )
          break;
        sub_729660(44);
      }
      sub_729660(41);
    }
    sub_729660(32);
    v3 = *(char **)(v1 + 16);
    v4 = *v3;
    if ( *v3 )
    {
      v5 = 0;
      while ( 1 )
      {
        v6 = v3 + 4;
        v7 = (unsigned __int8)v3[1]
           | ((unsigned __int64)(unsigned __int8)v3[3] << 16)
           | ((unsigned __int64)(unsigned __int8)v3[2] << 8);
        v8 = v7;
        switch ( v4 )
        {
          case 1:
            if ( v7 )
            {
              do
              {
                v13 = *v6;
                if ( *v6 )
                {
                  ++v6;
                  sub_729660(v13);
                }
                else
                {
                  if ( v6[1] == 6 )
                    sub_729660(0);
                  v6 += 2;
                  --v8;
                }
                --v8;
              }
              while ( v8 );
            }
            goto LABEL_9;
          case 2:
            sub_7295A0(" ## ");
            goto LABEL_9;
          case 3:
          case 8:
            v10 = *(_QWORD *)(v1 + 8);
            for ( j = v7 - 1; j; --j )
              v10 = *(_QWORD *)(v10 + 8);
            goto LABEL_15;
          case 4:
          case 5:
            v9 = "#";
            if ( v4 == 5 )
              v9 = "#@";
            sub_7295A0(v9);
            if ( v8 == 0xFFFFFF )
              goto LABEL_9;
            v10 = *(_QWORD *)(v1 + 8);
            v15 = v8 - 1;
            if ( v8 != 1 )
            {
              do
              {
                v10 = *(_QWORD *)(v10 + 8);
                --v15;
              }
              while ( v15 );
            }
LABEL_15:
            sub_7295A0(*(char **)v10);
            v4 = *v6;
            if ( !*v6 )
              return sub_729660(0);
            goto LABEL_10;
          case 6:
            v10 = *(_QWORD *)(v1 + 8);
            for ( k = v7 - 1; k; --k )
              v10 = *(_QWORD *)(v10 + 8);
            goto LABEL_15;
          case 7:
            goto LABEL_9;
          case 9:
            v5 = &v6[v7];
            sub_7295A0("__VA_OPT__(");
LABEL_9:
            v4 = *v6;
            if ( !*v6 )
              return sub_729660(0);
LABEL_10:
            if ( v5 == v6 )
            {
              sub_729660(41);
              v4 = *v5;
              v5 = 0;
            }
            v3 = v6;
            break;
          default:
            sub_721090();
        }
      }
    }
  }
  return sub_729660(0);
}

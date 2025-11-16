// Function: sub_819070
// Address: 0x819070
//
__int64 __fastcall sub_819070(char *a1)
{
  const char *v1; // rsi
  __int64 v2; // r13
  __int64 *v3; // rbx
  char *v4; // rdi
  unsigned int v5; // eax
  const char *v6; // r12
  unsigned int v7; // eax

  v1 = qword_4F06410;
  v2 = unk_4F06400;
  v3 = (__int64 *)unk_4F06458;
  if ( unk_4F06458 )
  {
    do
    {
      while ( 1 )
      {
        v6 = (const char *)v3[1];
        if ( (unsigned __int64)v6 > qword_4F06408 )
          goto LABEL_14;
        if ( v6 >= v1 )
          break;
LABEL_7:
        v3 = (__int64 *)*v3;
        if ( !v3 )
          goto LABEL_14;
      }
      if ( a1 )
      {
        v4 = (char *)memcpy(a1, v1, v6 - v1);
        v5 = *((_DWORD *)v3 + 4);
        a1 = &v4[v6 - v1];
        if ( v5 == 2 )
        {
          *a1 = 10;
          --v2;
          ++a1;
          v1 = v6 + 2;
        }
        else if ( v5 > 2 )
        {
          if ( v5 != 3 )
            goto LABEL_21;
          v1 = v6;
        }
        else
        {
          v2 += 2;
          if ( v5 )
          {
            a1 += 2;
            v1 = v6;
            *((_WORD *)a1 - 1) = 2652;
          }
          else
          {
            a1 += 3;
            *(_WORD *)(a1 - 3) = 16191;
            v1 = v6 + 1;
            *(a1 - 1) = *((_BYTE *)v3 + 24);
          }
        }
        goto LABEL_7;
      }
      v7 = *((_DWORD *)v3 + 4);
      if ( v7 == 2 )
      {
        --v2;
        goto LABEL_7;
      }
      if ( v7 > 2 )
      {
        if ( v7 != 3 )
LABEL_21:
          sub_721090();
        goto LABEL_7;
      }
      v3 = (__int64 *)*v3;
      v2 += 2;
    }
    while ( v3 );
  }
LABEL_14:
  if ( a1 && qword_4F06408 >= (unsigned __int64)v1 )
    memcpy(a1, v1, qword_4F06408 - (_QWORD)v1 + 1LL);
  return v2;
}

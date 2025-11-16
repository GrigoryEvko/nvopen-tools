// Function: sub_810E90
// Address: 0x810e90
//
void __fastcall sub_810E90(__int64 a1, char a2, char a3, unsigned __int8 a4, int a5, __int64 a6, char *s, __int64 a8)
{
  const char *v9; // r15
  size_t v10; // rax
  __int64 v11; // rdx
  size_t v12; // rdx
  int v13; // ebx
  char *v14; // r13
  size_t v15; // rdx
  const char *v16; // rsi
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // r13
  char *v20; // rsi
  _BYTE v21[112]; // [rsp+10h] [rbp-70h] BYREF

  if ( !a2 || a2 == 6 )
  {
    if ( (*(_BYTE *)(a1 + 89) & 8) != 0 )
    {
      v9 = *(const char **)(a1 + 24);
      v10 = strlen(v9);
      if ( v10 <= 9 )
        goto LABEL_6;
    }
    else
    {
      v9 = *(const char **)(a1 + 8);
      v10 = strlen(v9);
      if ( v10 <= 9 )
      {
LABEL_6:
        v21[1] = 0;
        v11 = 1;
        v21[0] = v10 + 48;
LABEL_7:
        *(_QWORD *)a8 += v11;
        sub_8238B0(qword_4F18BE0, v21, v11);
        v12 = strlen(v9);
LABEL_8:
        *(_QWORD *)a8 += v12;
        sub_8238B0(qword_4F18BE0, v9, v12);
        if ( a2 == 3 )
        {
LABEL_14:
          v13 = *(_DWORD *)(a8 + 52);
          *(_DWORD *)(a8 + 52) = 1;
          sub_80F5E0(a6, 0, (_QWORD *)a8);
          *(_DWORD *)(a8 + 52) = v13;
          return;
        }
        goto LABEL_9;
      }
    }
    v11 = (int)sub_622470(v10, v21);
    goto LABEL_7;
  }
  switch ( a2 )
  {
    case 1:
      if ( (*(_BYTE *)(a1 + 194) & 0x40) == 0 )
      {
        if ( a4 == 2 )
        {
          v15 = 2;
          v14 = "C2";
        }
        else
        {
          if ( a4 <= 2u )
          {
            v20 = "C1";
            goto LABEL_30;
          }
          if ( a4 != 4 )
            goto LABEL_43;
          v15 = 2;
          v14 = "C9";
        }
LABEL_17:
        *(_QWORD *)a8 += v15;
        sub_8238B0(qword_4F18BE0, v14, v15);
        goto LABEL_9;
      }
      if ( a4 != 1 )
      {
        if ( a4 == 2 )
        {
          v16 = "CI2";
          goto LABEL_22;
        }
LABEL_43:
        sub_721090();
      }
      v16 = "CI1";
      do
LABEL_22:
        a1 = *(_QWORD *)(a1 + 232);
      while ( (*(_BYTE *)(a1 + 194) & 0x40) != 0 );
      v17 = *(_QWORD *)(a1 + 40);
      v18 = qword_4F18BE0;
      v19 = *(_QWORD *)(v17 + 32);
      *(_QWORD *)a8 += 3LL;
      sub_8238B0(v18, v16, 3);
      if ( s )
        goto LABEL_10;
      if ( v19 )
        sub_810650(v19, 1, (_QWORD *)a8);
      return;
    case 2:
      if ( a4 == 3 )
      {
        v15 = 2;
        v14 = "D0";
      }
      else if ( a4 > 3u )
      {
        if ( a4 != 4 )
          goto LABEL_43;
        v15 = 2;
        v14 = "D9";
      }
      else
      {
        if ( a4 <= 1u )
        {
          v20 = "D1";
LABEL_30:
          *(_QWORD *)a8 += 2LL;
          sub_8238B0(qword_4F18BE0, v20, 2);
LABEL_9:
          if ( !s )
            return;
LABEL_10:
          sub_80BC40(s, (_QWORD *)a8);
          return;
        }
        v15 = 2;
        v14 = "D2";
      }
      goto LABEL_17;
    case 3:
      *(_QWORD *)a8 += 2LL;
      sub_8238B0(qword_4F18BE0, "cv", 2);
      goto LABEL_14;
    case 4:
      v12 = 2;
      v9 = "li";
      goto LABEL_8;
    case 5:
      v14 = sub_8094C0(a3, a5);
      v15 = strlen(v14);
      goto LABEL_17;
    default:
      goto LABEL_43;
  }
}

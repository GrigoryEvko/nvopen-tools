// Function: sub_668C50
// Address: 0x668c50
//
__int64 __fastcall sub_668C50(_WORD *a1, __int16 a2, int a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int16 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int16 v9; // bx
  __int64 v11; // rdx
  __int64 v12; // rcx
  char *v13; // rsi
  __int64 v14; // rdi
  unsigned int *v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // rcx
  int v23; // [rsp+0h] [rbp-80h]
  int v24; // [rsp+4h] [rbp-7Ch]
  _BYTE v26[32]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE v27[80]; // [rsp+30h] [rbp-50h] BYREF

  sub_7ADF70(v26, 1);
  sub_7AE360(v26);
  v6 = sub_7B8B50(v26, 1, v4, v5);
  if ( v6 != 1 )
  {
    sub_7AE210(v26);
    if ( a3 || v6 != a2 && v6 != 55 && v6 != 17 )
    {
LABEL_5:
      *a1 = v6;
      sub_7BC270(v26);
      return sub_7AEA70(v26);
    }
    v9 = v6;
    sub_7BC160(v26);
    *a1 = v6;
    sub_7ADF70(v27, 0);
    goto LABEL_8;
  }
  do
  {
    sub_7AE360(v26);
    v9 = sub_7B8B50(v26, 1, v7, v8);
  }
  while ( v9 == 1 );
  sub_7AE210(v26);
  if ( v9 != 55 && v9 != a2 && v9 != 17 )
    goto LABEL_5;
  sub_7BC160(v26);
  *a1 = v9;
  sub_7ADF70(v27, 0);
  if ( !a3 )
  {
LABEL_8:
    v23 = 0;
    if ( word_4F06418[0] == 9 )
    {
      sub_7B8B50(v27, 0, v11, v12);
      *a1 = v6;
      sub_7BC270(v26);
      return sub_7AEA70(v26);
    }
    goto LABEL_9;
  }
  sub_7AE360(v27);
  sub_7B8B50(v27, 0, v19, v20);
  v23 = 1;
  if ( word_4F06418[0] != 9 )
  {
LABEL_9:
    v24 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v13 = "final";
        v14 = 241;
        if ( (unsigned int)sub_7C8F50(241, "final") )
        {
LABEL_13:
          v14 = (__int64)v27;
          sub_7AE360(v27);
          v24 = 1;
          goto LABEL_14;
        }
        v15 = &dword_4F077BC;
        v16 = dword_4F077BC;
        if ( !dword_4F077BC )
          break;
        if ( !(_DWORD)qword_4F077B4 && qword_4F077A8 <= 0x9EFBu )
          goto LABEL_14;
LABEL_12:
        v13 = "__final";
        v14 = 241;
        if ( (unsigned int)sub_7C8F50(241, "__final") )
          goto LABEL_13;
LABEL_14:
        sub_7B8B50(v14, v13, v15, v16);
        if ( word_4F06418[0] == 9 )
          goto LABEL_19;
      }
      if ( (_DWORD)qword_4F077B4 )
        goto LABEL_12;
      sub_7B8B50(241, "final", &dword_4F077BC, dword_4F077BC);
      if ( word_4F06418[0] == 9 )
      {
LABEL_19:
        sub_7B8B50(v14, v13, v17, v18);
        if ( v24 )
        {
          *a1 = v9;
          sub_7BC000(v27);
        }
        else
        {
          *a1 = v6;
          sub_7BC270(v26);
          if ( v23 )
            goto LABEL_21;
        }
        return sub_7AEA70(v26);
      }
    }
  }
  sub_7B8B50(v27, 0, v21, v22);
  *a1 = 1;
  sub_7BC270(v26);
LABEL_21:
  sub_7AEA70(v27);
  return sub_7AEA70(v26);
}

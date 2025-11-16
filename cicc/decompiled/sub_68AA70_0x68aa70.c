// Function: sub_68AA70
// Address: 0x68aa70
//
__int64 __fastcall sub_68AA70(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  __int64 result; // rax
  char *v6; // r13
  _DWORD *v7; // rax
  char *v8; // rdi
  __int64 v9; // rax
  _BYTE *v10; // rax
  __int64 v11; // [rsp+0h] [rbp-C0h]
  __int64 v13; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v14; // [rsp+18h] [rbp-A8h] BYREF
  char s[160]; // [rsp+20h] [rbp-A0h] BYREF

  v14 = a1;
  v13 = a2;
  result = sub_89A1A0(a2, a1, &v13, &v14);
  if ( v14 )
  {
    while ( 1 )
    {
      v10 = *(_BYTE **)(v13 + 8);
      v6 = *(char **)(*(_QWORD *)v10 + 8LL);
      if ( !dword_4F077BC )
        break;
      if ( strcmp(*(const char **)(*(_QWORD *)v10 + 8LL), "<unnamed>") )
      {
        if ( !*a3 )
        {
LABEL_8:
          v8 = "; ";
          if ( (_DWORD)qword_4F077B4 )
            goto LABEL_9;
          goto LABEL_10;
        }
        goto LABEL_19;
      }
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( v10[80] == 2 )
        {
          v6 = "<anonymous>";
          v8 = "; ";
          if ( !*a3 )
            goto LABEL_10;
        }
        else
        {
          v6 = s;
          v7 = (_DWORD *)sub_892BC0(v13);
          sprintf(s, "<template-parameter-%d-%d>", v7[1], *v7);
          if ( !*a3 )
          {
            if ( dword_4F077BC )
              goto LABEL_8;
LABEL_9:
            v8 = ", ";
LABEL_10:
            sub_7295A0(v8);
LABEL_11:
            if ( dword_4F077BC && !(_DWORD)qword_4F077B4 && (v9 = *(_QWORD *)(v13 + 8), *(_BYTE *)(v9 + 80) == 2) )
            {
              v11 = *(_QWORD *)(*(_QWORD *)(v9 + 88) + 128LL);
              sub_74A390(v11, 0, 1, 0, 0, a4);
              sub_7295A0(v6);
              if ( v11 )
                sub_74D110(v11, 0, 0, a4);
            }
            else
            {
              sub_7295A0(v6);
            }
            sub_7295A0(" = ");
            sub_747370(v14, a4);
            goto LABEL_16;
          }
        }
LABEL_19:
        sub_7295A0(" [with ");
        *a3 = 0;
        goto LABEL_11;
      }
LABEL_16:
      result = sub_89A1C0(&v13, &v14);
      if ( !v14 )
        return result;
    }
    if ( !*a3 )
      goto LABEL_9;
    goto LABEL_19;
  }
  return result;
}

// Function: sub_C85FC0
// Address: 0xc85fc0
//
char *__fastcall sub_C85FC0(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  char *v5; // r12
  __int64 v6; // rdx
  char v7; // al
  __int64 v9; // r13
  int *v10; // rax
  __int64 v11; // rax
  __int64 v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v5 = a1;
  if ( !byte_4F84120 )
  {
    a1 = &byte_4F84120;
    if ( (unsigned int)sub_2207590(&byte_4F84120) )
    {
      a1 = &byte_4F84120;
      dword_4F84128 = getpagesize();
      sub_2207640(&byte_4F84120);
      v6 = (unsigned int)dword_4F84128;
      if ( dword_4F84128 == -1 )
        goto LABEL_6;
LABEL_3:
      v7 = v5[8];
      *(_DWORD *)v5 = v6;
      v5[8] = v7 & 0xFC | 2;
      return v5;
    }
  }
  v6 = (unsigned int)dword_4F84128;
  if ( dword_4F84128 != -1 )
    goto LABEL_3;
LABEL_6:
  v9 = sub_2241E50(a1, a2, v6, a4, a5);
  v10 = __errno_location();
  sub_C63CA0(v12, *v10, v9);
  v11 = v12[0];
  v5[8] |= 3u;
  *(_QWORD *)v5 = v11 & 0xFFFFFFFFFFFFFFFELL;
  return v5;
}

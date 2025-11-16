// Function: sub_12BB400
// Address: 0x12bb400
//
__int64 __fastcall sub_12BB400(__int64 a1, __int64 a2, __int64 a3, const char *a4)
{
  char v6; // bl
  unsigned int v7; // r8d
  unsigned int v9; // [rsp+4h] [rbp-4Ch]
  __int64 v10; // [rsp+8h] [rbp-48h]
  unsigned int v11[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v6 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v10 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a2 )
    {
      v7 = 4;
      goto LABEL_16;
    }
    v6 = 1;
LABEL_11:
    if ( (unsigned __int8)sub_1C17C80(a2, a3) > 7u )
    {
LABEL_12:
      v7 = sub_2258EF0(a1, a2, a3, a4);
      goto LABEL_13;
    }
    v11[0] = 0;
    v7 = sub_12BB290(a2, a3, a4, (int *)v11);
    if ( !v7 )
    {
      if ( sub_12B9F70(v11[0]) )
        goto LABEL_12;
      v7 = sub_12BE5C0(a1, a2, a3, a4);
    }
LABEL_13:
    if ( !v6 )
      return v7;
LABEL_16:
    v9 = v7;
    sub_16C30E0(v10);
    return v9;
  }
  if ( !qword_4F92D80 )
    sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
  v10 = qword_4F92D80;
  if ( a2 )
    goto LABEL_11;
  return 4;
}

// Function: sub_16893B0
// Address: 0x16893b0
//
int __fastcall sub_16893B0(char *format, __gnuc_va_list arg)
{
  __int64 v4; // r14
  __int64 v5; // rdi
  int v6; // edx
  int v7; // ecx
  int v8; // r8d
  int v9; // r9d
  char *v10; // r13
  const char *v11; // rsi
  char *v12; // r12
  __int64 **v13; // rax
  _QWORD **v14; // rax
  __int64 v15; // rdx
  _BYTE *v16; // r12
  void (__fastcall *v17)(_BYTE *); // rax
  _QWORD *v18; // rdi
  char *v19; // rbx
  __int64 v20; // rdx
  __int64 **v21; // rax
  _QWORD *v23; // r15
  FILE *v24; // rdi
  char v25; // [rsp+0h] [rbp-30h]

  if ( sub_16893A0() )
  {
    v4 = sub_1683C60(0);
    v5 = *((_QWORD *)sub_1689050() + 3);
    v10 = (char *)sub_1685080(v5, 100000);
    if ( !v10 )
      sub_1683C30(v5, 100000, v6, v7, v8, v9, v25);
    v11 = format;
    v12 = v10;
    vsprintf(v10, v11, arg);
    if ( *v10 )
    {
      while ( 1 )
      {
        v19 = strchr(v12, 10);
        if ( !*((_QWORD *)sub_1689050() + 11) )
        {
          v23 = sub_1688290(128, 10, v20);
          *((_QWORD *)sub_1689050() + 11) = v23;
        }
        if ( !v19 )
          break;
        *v19 = 0;
        v13 = (__int64 **)sub_1689050();
        sub_16884F0(v13[11], v12);
        v14 = (_QWORD **)sub_1689050();
        v16 = sub_16884C0(v14[11], (__int64)v12, v15);
        *((_QWORD *)sub_1689050() + 11) = 0;
        if ( sub_16893A0() )
        {
          v17 = (void (__fastcall *)(_BYTE *))sub_16893A0();
          v17(v16);
        }
        v18 = v16;
        v12 = v19 + 1;
        sub_16856A0(v18);
        if ( !v19[1] )
          goto LABEL_13;
      }
      v21 = (__int64 **)sub_1689050();
      sub_16884F0(v21[11], v12);
    }
LABEL_13:
    sub_16856A0(v10);
    return sub_1683C60(v4);
  }
  else
  {
    v24 = qword_4F9F878;
    if ( !qword_4F9F878 )
      v24 = stderr;
    return vfprintf(v24, format, arg);
  }
}

// Function: sub_12BB290
// Address: 0x12bb290
//
__int64 __fastcall sub_12BB290(__int64 a1, __int64 a2, const char *a3, int *a4)
{
  char v5; // r14
  __int64 v6; // r15
  unsigned int v7; // r12d
  size_t v8; // rax
  const char *v9; // rcx
  __int64 v10; // r13
  int v11; // eax
  _QWORD v14[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v6 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v7 = 4;
LABEL_20:
      sub_16C30E0(v6);
      return v7;
    }
    v5 = 1;
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v6 = qword_4F92D80;
    if ( !a1 )
      return 4;
  }
  if ( !a3 )
    a3 = "<unnamed>";
  v8 = strlen(a3);
  v9 = a3;
  v7 = 4;
  sub_16C2450(v14, a1, a2, v9, v8, 0);
  v10 = v14[0];
  if ( v14[0] )
  {
    v11 = sub_1C1ADE0(v14[0]);
    if ( v11 )
    {
      v7 = 0;
      *a4 = v11 / 10;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v10 + 8LL))(v10);
  }
  if ( v5 )
    goto LABEL_20;
  return v7;
}

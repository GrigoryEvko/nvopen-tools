// Function: sub_12BA6A0
// Address: 0x12ba6a0
//
__int64 __fastcall sub_12BA6A0(__int64 a1, _DWORD *a2)
{
  char v2; // r13
  __int64 v3; // r14
  unsigned int v4; // r12d
  __int64 v5; // rax

  v2 = byte_4F92D70;
  if ( byte_4F92D70 || !dword_4C6F008 )
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v3 = qword_4F92D80;
    sub_16C30C0(qword_4F92D80);
    if ( !a1 )
    {
      v4 = 5;
      goto LABEL_18;
    }
    v2 = 1;
  }
  else
  {
    if ( !qword_4F92D80 )
      sub_16C1EA0(&qword_4F92D80, sub_12B9A60, sub_12B9AC0);
    v3 = qword_4F92D80;
    if ( !a1 )
      return 5;
  }
  v5 = (__int64)(*(_QWORD *)(a1 + 192) - *(_QWORD *)(a1 + 184)) >> 3;
  if ( !(_DWORD)v5 )
    LODWORD(v5) = 1;
  if ( a2 )
  {
    *a2 = v5;
    v4 = 0;
  }
  else
  {
    v4 = 4;
  }
  if ( !v2 )
    return v4;
LABEL_18:
  sub_16C30E0(v3);
  return v4;
}

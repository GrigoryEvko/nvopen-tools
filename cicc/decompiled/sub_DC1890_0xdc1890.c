// Function: sub_DC1890
// Address: 0xdc1890
//
__int64 __fastcall sub_DC1890(__int64 a1, __int64 a2, int a3)
{
  _BOOL4 v3; // ebx
  unsigned int v4; // r13d
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  char v8; // al
  __int64 v10; // r15
  __int64 v11; // [rsp+8h] [rbp-38h]

  if ( (unsigned int)(a3 - 32) <= 1 )
  {
    BYTE4(v11) = 0;
    return v11;
  }
  v4 = (a3 & 0xFFFFFFFB) - 34;
  if ( !sub_B532A0(a3) )
  {
    if ( (*(_BYTE *)(a2 + 28) & 4) == 0 )
      goto LABEL_4;
    v10 = sub_D33D80((_QWORD *)a2, a1, v5, v6, v7);
    if ( !(unsigned __int8)sub_DBED40(a1, v10) )
    {
      if ( (unsigned __int8)sub_DBEC80(a1, v10) )
      {
        v3 = v4 <= 1;
        v8 = 1;
        goto LABEL_5;
      }
      goto LABEL_4;
    }
LABEL_9:
    v8 = 1;
    v3 = v4 > 1;
    goto LABEL_5;
  }
  if ( (*(_BYTE *)(a2 + 28) & 2) != 0 )
    goto LABEL_9;
LABEL_4:
  v8 = 0;
LABEL_5:
  LODWORD(v11) = v3;
  BYTE4(v11) = v8;
  return v11;
}

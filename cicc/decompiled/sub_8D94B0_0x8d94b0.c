// Function: sub_8D94B0
// Address: 0x8d94b0
//
__int64 __fastcall sub_8D94B0(
        __int64 *a1,
        __int64 (__fastcall *a2)(__int64, unsigned int *),
        __int64 a3,
        unsigned int a4)
{
  char v4; // al
  unsigned int v7; // r12d
  __int64 v8; // rdi
  __int64 v10; // rax
  __int64 *v11[7]; // [rsp+8h] [rbp-38h] BYREF

  v11[0] = a1;
  if ( !a1 )
    return 0;
  v4 = *((_BYTE *)a1 + 8);
  if ( v4 != 3 )
    goto LABEL_3;
  sub_72F220(v11);
  a1 = v11[0];
  if ( !v11[0] )
    return 0;
  v4 = *((_BYTE *)v11[0] + 8);
LABEL_3:
  v7 = 0;
LABEL_4:
  if ( v4 )
  {
LABEL_5:
    if ( v4 == 2 )
    {
      if ( !v7 )
      {
        v10 = a1[4];
        if ( (*(_BYTE *)(v10 + 89) & 4) != 0 )
        {
          v7 = a4 & 0x200;
          if ( (a4 & 0x200) != 0 )
            v7 = sub_8D8C50(*(_QWORD *)(*(_QWORD *)(v10 + 40) + 32LL), a2, a3, a4);
        }
      }
    }
    else if ( (a1[3] & 1) == 0 )
    {
      v8 = a1[4];
      if ( v8 )
      {
        if ( (a4 & 0x2100) != 0x100 )
          v7 = sub_8D93A0(v8, a2, a3, a4);
      }
    }
    goto LABEL_10;
  }
  while ( !(unsigned int)sub_8D8C50(a1[4], a2, a3, a4) )
  {
LABEL_10:
    a1 = (__int64 *)*v11[0];
    v11[0] = a1;
    if ( !a1 )
      return v7;
    v4 = *((_BYTE *)a1 + 8);
    if ( v4 != 3 )
      goto LABEL_4;
    sub_72F220(v11);
    a1 = v11[0];
    if ( !v11[0] )
      return v7;
    v4 = *((_BYTE *)v11[0] + 8);
    if ( v4 )
      goto LABEL_5;
  }
  return 1;
}

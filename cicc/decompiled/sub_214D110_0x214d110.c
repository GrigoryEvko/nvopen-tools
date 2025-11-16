// Function: sub_214D110
// Address: 0x214d110
//
char __fastcall sub_214D110(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // r9
  unsigned __int64 v6; // rcx
  char result; // al
  char v8; // [rsp+Fh] [rbp-1h]

  v5 = *(_QWORD *)(a2 + 32) + 40LL * a3;
  v6 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL);
  if ( (v6 & 0x80u) != 0LL )
  {
    result = 0;
    if ( a3 != 4 )
    {
      if ( a3 != 5 || *(_BYTE *)v5 != 1 || (v6 & 0x1000) != 0 )
        return result;
LABEL_10:
      sub_214CDF0(a1, *(_QWORD *)(v5 + 24), a4);
      return 1;
    }
LABEL_9:
    if ( *(_BYTE *)v5 != 1 )
      return result;
    goto LABEL_10;
  }
  if ( (v6 & 0x300) != 0 )
  {
    result = 0;
    if ( a3 != 1 << ((BYTE1(v6) & 3) - 1) )
      return result;
    goto LABEL_9;
  }
  if ( (v6 & 0x400) != 0 )
  {
    result = 0;
    if ( a3 )
      return result;
    goto LABEL_9;
  }
  result = (v6 >> 11) & (a3 == 1);
  if ( result )
  {
    if ( *(_BYTE *)v5 == 1 )
    {
      v8 = (v6 >> 11) & (a3 == 1);
      sub_214CDF0(a1, *(_QWORD *)(v5 + 24), a4);
      return v8;
    }
    else
    {
      return 0;
    }
  }
  return result;
}

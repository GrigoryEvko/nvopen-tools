// Function: sub_6E2B30
// Address: 0x6e2b30
//
__int64 *__fastcall sub_6E2B30(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 *v5; // rcx
  __int64 *result; // rax
  __int64 v7; // rdx

  if ( *(_QWORD *)(qword_4D03C50 + 48LL) )
  {
    if ( *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u )
      sub_6E2B00(a1, a2);
    sub_733F40(0);
  }
  v2 = unk_4D03C48;
  for ( unk_4D03C48 = 0; v2; *(_QWORD *)(v4 + 40) = v3 )
  {
    v4 = v2;
    v2 = *(_QWORD *)(v2 + 40);
    if ( !*(_BYTE *)(v4 + 8) )
    {
      sub_875E10(*(_QWORD *)v4, *(_QWORD *)(v4 + 16), v4 + 32, 1, *(_QWORD *)(v4 + 24));
      *(_BYTE *)(v4 + 8) = 1;
    }
    v3 = qword_4D03A98;
    qword_4D03A98 = v4;
  }
  v5 = (__int64 *)qword_4D03C50;
  unk_4D03C48 = *(_QWORD *)(qword_4D03C50 + 8LL);
  if ( (*(_DWORD *)(qword_4D03C50 + 16LL) & 0x2002FF) == 5 )
  {
    sub_6E24C0();
    v5 = (__int64 *)qword_4D03C50;
  }
  result = (__int64 *)v5[18];
  if ( result && (*((_BYTE *)v5 + 19) & 1) != 0 )
    *((_BYTE *)result + 56) = 1;
  v7 = *v5;
  if ( *v5 )
  {
    result = *(__int64 **)(v7 + 120);
    if ( result )
    {
      while ( 1 )
      {
        result = (__int64 *)*result;
        if ( !result )
          break;
        *(_QWORD *)(v7 + 120) = result;
      }
      if ( (*((_BYTE *)v5 + 19) & 1) != 0 )
      {
LABEL_18:
        if ( (*(_BYTE *)(v7 + 18) & 0x40) != 0 )
          *(_BYTE *)(v7 + 19) |= 1u;
      }
    }
    else if ( (*((_BYTE *)v5 + 19) & 1) != 0 )
    {
      goto LABEL_18;
    }
    if ( (*((_BYTE *)v5 + 21) & 0x40) != 0 )
      *(_BYTE *)(v7 + 21) |= 0x40u;
  }
  qword_4D03C50 = v7;
  return result;
}

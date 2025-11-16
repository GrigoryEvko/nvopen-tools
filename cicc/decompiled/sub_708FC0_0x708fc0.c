// Function: sub_708FC0
// Address: 0x708fc0
//
__int64 __fastcall sub_708FC0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // rcx
  __int64 result; // rax

  v1 = unk_4D03FE8;
  if ( unk_4D03FE8 )
    sub_7F4B80();
  sub_72AC40();
  v4 = unk_4D03FE8;
  if ( unk_4D03FE8 )
  {
    if ( dword_4F077C4 == 2 )
    {
      a1 = 0;
      sub_733F40(0);
      if ( (unsigned int)sub_7E16F0() )
      {
        a1 = *(_QWORD *)(unk_4D03FF0 + 8LL);
        sub_7E9A00(a1);
      }
    }
  }
  sub_863FC0(a1, v1, v2, v4, v3);
  result = sub_825190();
  if ( unk_4D03FE8 )
  {
    if ( unk_4F074B0 || (sub_8622C0(*(_QWORD *)(unk_4D03FF0 + 8LL)), a1 = *(_QWORD *)(unk_4D03FF0 + 8LL), unk_4F074B0) )
    {
      if ( (unsigned int)sub_7E16F0() )
        goto LABEL_15;
    }
    else
    {
      unk_4D03B60 = 1;
      sub_75B260(a1, 23);
      unk_4D03B60 = 0;
      if ( (unsigned int)sub_7E16F0() )
      {
LABEL_15:
        sub_80D280();
        sub_708F30();
        if ( (unsigned int)sub_7E16F0() )
        {
LABEL_9:
          if ( unk_4D03F84 )
            sub_737180(a1);
        }
LABEL_11:
        nullsub_15();
        if ( (unsigned int)sub_7E16F0() )
          sub_7E98C0();
        return sub_823780(unk_4F073B8);
      }
    }
    sub_708F30();
    if ( (unsigned int)sub_7E16F0() )
      goto LABEL_9;
    goto LABEL_11;
  }
  return result;
}

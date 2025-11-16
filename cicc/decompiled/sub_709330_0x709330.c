// Function: sub_709330
// Address: 0x709330
//
_QWORD *sub_709330()
{
  _QWORD *v0; // rdi
  _QWORD *v1; // r12
  __int64 v2; // rdi
  _QWORD *v3; // r13
  _QWORD *v4; // r15
  __int64 v5; // rdi
  _QWORD *v6; // r13
  _QWORD *v7; // r13
  __int64 v8; // rbx
  int v9; // r13d
  __int64 v10; // rdi
  unsigned int v12; // r13d
  int v13; // r15d

  sub_8D0910(qword_4D03FD0);
  if ( dword_4F077C4 == 2 )
  {
    unk_4D045D8("Generating Needed Template Instantiations", byte_3F871B3);
    sub_8B18F0();
    unk_4D045D0();
    if ( !unk_4D04910 )
      goto LABEL_3;
  }
  else if ( !unk_4D04910 )
  {
    goto LABEL_3;
  }
  sub_81FFB0();
LABEL_3:
  unk_4D045D8("Wrapping up translation unit", byte_3F871B3);
  v0 = qword_4D03FD0;
  v1 = (_QWORD *)*qword_4D03FD0;
  if ( *qword_4D03FD0 )
  {
    do
    {
      sub_8D0910(v1);
      sub_709140((__int64)v1, byte_3F871B3);
      v1 = (_QWORD *)*v1;
    }
    while ( v1 );
    v0 = qword_4D03FD0;
  }
  sub_8D0910(v0);
  sub_709140((__int64)v0, byte_3F871B3);
  if ( !unk_4F074B0 )
    sub_8C6110();
  v2 = (__int64)qword_4D03FD0;
  v3 = (_QWORD *)*qword_4D03FD0;
  if ( *qword_4D03FD0 )
  {
    do
    {
      while ( 1 )
      {
        sub_8D0910(v3);
        if ( !unk_4F074B0 )
          break;
        v3 = (_QWORD *)*v3;
        if ( !v3 )
          goto LABEL_13;
      }
      sub_8622C0(*(_QWORD *)(unk_4D03FF0 + 8LL));
      v3 = (_QWORD *)*v3;
    }
    while ( v3 );
LABEL_13:
    v2 = (__int64)qword_4D03FD0;
    v4 = (_QWORD *)*qword_4D03FD0;
    if ( *qword_4D03FD0 )
    {
      do
      {
        while ( 1 )
        {
          sub_8D0910(v4);
          if ( !unk_4F074B0 )
            break;
          v4 = (_QWORD *)*v4;
          if ( !v4 )
            goto LABEL_18;
        }
        v5 = *(_QWORD *)(unk_4D03FF0 + 8LL);
        unk_4D03B60 = 1;
        sub_75B260(v5, 23);
        unk_4D03B60 = 0;
        v4 = (_QWORD *)*v4;
      }
      while ( v4 );
LABEL_18:
      v2 = (__int64)qword_4D03FD0;
      v6 = (_QWORD *)*qword_4D03FD0;
      if ( *qword_4D03FD0 )
      {
        do
        {
          sub_8D0910(v6);
          sub_708F30();
          v6 = (_QWORD *)*v6;
        }
        while ( v6 );
        v2 = (__int64)qword_4D03FD0;
        v7 = (_QWORD *)*qword_4D03FD0;
        if ( *qword_4D03FD0 )
        {
          do
          {
            sub_8D0910(v7);
            sub_708FC0((__int64)v7);
            v7 = (_QWORD *)*v7;
          }
          while ( v7 );
          v2 = (__int64)qword_4D03FD0;
        }
      }
    }
  }
  if ( !unk_4F074B0 && !(unk_4D048F8 | unk_4D04958) && *(_QWORD *)v2 )
  {
    sub_8C5CD0();
    v2 = (__int64)qword_4D03FD0;
    unk_4F04C20 = 1;
  }
  sub_8D0910(v2);
  if ( unk_4F04C20 )
  {
    if ( unk_4F073A0 > 0 )
    {
      v12 = 1;
      do
      {
        v13 = 1;
        do
        {
          v2 = (unsigned int)v13++;
          sub_862730(v2, v12);
        }
        while ( unk_4F073A0 >= v13 );
        if ( !v12 )
          break;
        v12 = 0;
      }
      while ( unk_4F073A0 > 0 );
    }
    unk_4F04C20 = 0;
  }
  if ( unk_4D04734 != 1 )
    sub_895FB0();
  sub_708FC0(v2);
  if ( *qword_4D03FD0 && unk_4F073A8 > 1 )
  {
    v8 = 16;
    v9 = 2;
    do
    {
      while ( !*(_QWORD *)(unk_4F073B0 + v8) || *(_BYTE *)(*(_QWORD *)(unk_4F072B0 + v8) + 28LL) )
      {
        ++v9;
        v8 += 8;
        if ( unk_4F073A8 < v9 )
          goto LABEL_35;
      }
      v10 = (unsigned int)v9;
      v8 += 8;
      ++v9;
      sub_823310(v10);
    }
    while ( unk_4F073A8 >= v9 );
  }
LABEL_35:
  unk_4D045D0();
  sub_686700(&qword_4D04928, 1513);
  sub_686700(&qword_4D04908, 1514);
  sub_686700(&qword_4D04900, 1515);
  qword_4F061C8 = 0;
  sub_67D0D0();
  unk_4D03FF0 = 0;
  dword_4F07588 = 0;
  qword_4F60220 = 0;
  qword_4F60228 = 0;
  return &qword_4F60228;
}

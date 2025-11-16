// Function: sub_6158E0
// Address: 0x6158e0
//
unsigned int *sub_6158E0()
{
  char v0; // r13
  char v1; // bl
  int v2; // r12d
  char v3; // bl
  unsigned int *result; // rax
  unsigned __int8 v5; // cl
  char v6; // al

  v0 = HIBYTE(word_4CF812A);
  v1 = word_4CF812A;
  v2 = dword_4F077C4;
  if ( !word_4CF812A )
  {
    dword_4F077B8 = 1;
    if ( dword_4F077C4 == 2 )
    {
      dword_4F077BC = 1;
      goto LABEL_18;
    }
    dword_4F077C0 = 1;
  }
  if ( !dword_4F077C4 )
  {
    if ( unk_4F07778 <= 199900 )
    {
      if ( unk_4D0436C )
        sub_614540();
    }
    else if ( unk_4D0436C )
    {
      if ( byte_4CF80F3 )
        goto LABEL_43;
      unk_4D0436C = 0;
    }
    goto LABEL_7;
  }
LABEL_18:
  if ( unk_4D0436C )
  {
    if ( byte_4CF80F3 )
      sub_6849E0(675);
    unk_4D0436C = 0;
  }
  sub_614540();
  if ( dword_4F077C0 )
  {
    if ( HIBYTE(word_4CF812F) | (unsigned __int8)(word_4CF812F | v1 | byte_4CF812C) )
      goto LABEL_43;
    dword_4F077C0 = 0;
    dword_4F077B8 = 0;
  }
  if ( v2 == 2 )
  {
    if ( !dword_4D04964 )
      goto LABEL_39;
    goto LABEL_24;
  }
LABEL_7:
  if ( qword_4D0495C )
  {
    if ( word_4CF806C )
      goto LABEL_43;
    qword_4D0495C = 0;
  }
  if ( dword_4F077BC )
  {
    if ( HIBYTE(word_4CF812F) | (unsigned __int8)(word_4CF812F | v0 | byte_4CF812C) )
      goto LABEL_43;
    dword_4F077BC = 0;
    dword_4F077B8 = 0;
  }
  if ( (unsigned __int8)byte_4CF816F
     | (unsigned __int8)(byte_4CF816E | byte_4CF8169 | byte_4CF8163 | byte_4CF813F | byte_4CF8157) )
  {
LABEL_43:
    sub_6849E0(1027);
  }
  if ( !dword_4D04964 )
    goto LABEL_39;
  if ( v2 == 1 )
    sub_6849E0(591);
LABEL_24:
  if ( qword_4D0495C )
  {
    if ( word_4CF806C )
      sub_6849E0(592);
    qword_4D0495C = 0;
  }
  if ( unk_4D0436C )
  {
    if ( byte_4CF80F3 )
      sub_6849E0(677);
    unk_4D0436C = 0;
  }
  if ( dword_4F077C0 )
  {
    v3 = HIBYTE(word_4CF812F) | word_4CF812F | byte_4CF812C | v1;
    if ( v3 )
      goto LABEL_43;
    dword_4F077C0 = 0;
    result = &dword_4F077B8;
    dword_4F077B8 = 0;
    if ( !dword_4F077BC )
      return result;
    v6 = 0;
    v5 = 0;
    goto LABEL_47;
  }
  if ( dword_4F077BC )
  {
    v3 = byte_4CF812C;
    v5 = word_4CF812F;
    v6 = HIBYTE(word_4CF812F);
LABEL_47:
    if ( !(v5 | (unsigned __int8)(v6 | v3 | v0)) )
    {
      dword_4F077BC = 0;
      dword_4F077B8 = 0;
      return &dword_4F077B8;
    }
    goto LABEL_43;
  }
LABEL_39:
  result = &dword_4F077B8;
  if ( dword_4F077B8 && unk_4D0436C )
  {
    if ( !byte_4CF80F3 )
    {
      unk_4D0436C = 0;
      return result;
    }
    goto LABEL_43;
  }
  return result;
}

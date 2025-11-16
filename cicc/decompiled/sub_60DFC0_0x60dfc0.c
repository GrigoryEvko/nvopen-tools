// Function: sub_60DFC0
// Address: 0x60dfc0
//
unsigned __int64 sub_60DFC0()
{
  unsigned __int64 result; // rax
  _BOOL4 v1; // r8d
  _BOOL4 v2; // edi
  unsigned int v3; // esi
  int v4; // edx

  result = qword_4F077A8;
  if ( dword_4F077B4 )
  {
    if ( byte_4CF812C )
    {
      v1 = qword_4F077A8 > 0x9D6Bu;
      v2 = (unsigned __int64)(qword_4F077A8 - 40000LL) <= 0x12B;
    }
    else if ( qword_4F077A8 > 0x9F5Fu )
    {
      v2 = 0;
      v1 = 1;
    }
    else
    {
      qword_4F077A8 = 40800;
      v2 = 0;
      v1 = 1;
      result = 40800;
    }
    unk_4F068D0 = 1;
    unk_4F068C8 = unk_4F077A0;
    dword_4F068C4 = 1;
  }
  else
  {
    v1 = qword_4F077A8 > 0x9D6Bu;
    unk_4F068D8 = qword_4F077A8;
    dword_4F068C4 = 1;
    v2 = (unsigned __int64)(qword_4F077A8 - 40000LL) <= 0x12B;
  }
  unk_4D04250 = result;
  if ( !byte_4CF811D )
  {
    unk_4D04798 = 1;
    unk_4D04794 = 1;
  }
  if ( !byte_4CF8121 )
    unk_4D0477C = 1;
  if ( !byte_4CF811F )
  {
    unk_4D04788 = 1;
    unk_4D04780 = 1;
  }
  unk_4D04784 = 1;
  if ( !byte_4CF8073 )
    unk_4D04748 = 1;
  unk_4F068FC = 0;
  unk_4F068F0 = 0;
  unk_4D044C0 = 1;
  unk_4D042A4 = 1;
  unk_4D042A8 = 1;
  if ( !byte_4CF80F0 )
    unk_4D04388 = 1;
  if ( !byte_4CF8138 )
    unk_4D04384 = 0;
  if ( !byte_4CF8137 )
    unk_4D04224 = result > 0x765B;
  unk_4D04298 = 1;
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || (v3 = dword_4F07774) != 0 )
      v3 = result > 0x9EFB;
  }
  else
  {
    v3 = unk_4F07778 > 199900;
  }
  unk_4D04294 = v3;
  unk_4D04280 = 1;
  unk_4D0427C = v1;
  unk_4D04328 = 1;
  HIDWORD(qword_4D043AC) = 1;
  if ( !byte_4CF813A )
    unk_4D042F8 = 0;
  unk_4F07768 = 1;
  if ( !byte_4CF8145 )
    unk_4D0420C = 1;
  unk_4D04240 = v2;
  dword_4D043E0 = 1;
  if ( !byte_4CF8150 )
    unk_4D044BC = 1;
  unk_4D04290 = 1;
  unk_4D04204 = 0;
  if ( !word_4CF812F )
  {
    dword_4F077B4 = 0;
    if ( result <= 0x9C3F )
    {
      unk_4D048BC = 0;
      v4 = unk_4D0455C;
      if ( result <= 0x76BF )
        goto LABEL_31;
LABEL_52:
      if ( !v4 )
        goto LABEL_34;
      if ( unk_4D04600 <= 0x334B3u )
      {
LABEL_33:
        dword_4D04284 = 0;
        goto LABEL_34;
      }
LABEL_65:
      if ( !unk_4F06A80 )
        goto LABEL_33;
LABEL_34:
      if ( result > 0x1D4BF )
      {
        result = (unsigned __int64)&dword_4D04188;
        dword_4D04188 = 1;
      }
LABEL_36:
      if ( byte_4CF8182 )
        return result;
LABEL_60:
      dword_4D0419C = 1;
      return (unsigned __int64)&dword_4D0419C;
    }
    v4 = unk_4D0455C;
LABEL_50:
    if ( result > 0x1387F )
      unk_4D041B8 = 1;
    goto LABEL_52;
  }
  if ( result <= 0x9C3F )
  {
    unk_4D048BC = 0;
    if ( dword_4F077B4 )
      goto LABEL_40;
LABEL_49:
    v4 = unk_4D0455C;
    if ( result <= 0x76BF )
    {
LABEL_31:
      unk_4D04288 = 0;
      dword_4D04284 = 0;
      if ( !v4 )
        goto LABEL_36;
      if ( unk_4D04600 <= 0x334B3u )
        goto LABEL_33;
      goto LABEL_65;
    }
    goto LABEL_50;
  }
  if ( !dword_4F077B4 )
    goto LABEL_49;
LABEL_40:
  result = unk_4F077A0;
  if ( unk_4F077A0 <= 0x765Bu || (unk_4D044C4 = 1, unk_4F077A0 <= 0x77EBu) )
  {
    unk_4D04288 = 0;
  }
  else
  {
    unk_4D0431C = 1;
    unk_4D04288 = 0;
    if ( unk_4F077A0 > 0x78B3u )
    {
      if ( unk_4F077A0 > 0xEA5Fu )
      {
        unk_4D0428C = 1;
        if ( unk_4F077A0 > 0x1D4BFu )
        {
          unk_4D041B8 = 1;
          if ( unk_4F077A0 > 0x1FBCFu )
          {
            dword_4D04188 = 1;
            if ( unk_4F077A0 > 0x222DFu )
              return result;
          }
        }
      }
      goto LABEL_59;
    }
  }
  result = (unsigned __int64)&dword_4D04284;
  dword_4D04284 = 0;
LABEL_59:
  if ( !byte_4CF8182 )
    goto LABEL_60;
  return result;
}

// Function: sub_7B39D0
// Address: 0x7b39d0
//
unsigned __int64 __fastcall sub_7B39D0(unsigned __int64 *a1, int a2, int a3, int a4)
{
  unsigned __int8 *v4; // r15
  int v5; // eax
  unsigned __int64 v6; // r12
  unsigned __int64 v7; // r14
  int v8; // r13d
  int v9; // edx
  unsigned int v10; // r13d
  unsigned __int64 v13; // [rsp+10h] [rbp-50h]
  int v14; // [rsp+1Ch] [rbp-44h]
  __int64 v15; // [rsp+28h] [rbp-38h]

  v13 = *a1;
  v4 = (unsigned __int8 *)(*a1 + 2);
  v5 = 0;
  if ( !(dword_4D03CB0[0] | unk_4D03CE4) )
    v5 = a4;
  v14 = v5;
  v6 = 0;
  v15 = *a1 + 4LL * (*(_BYTE *)(*a1 + 1) != 117) + 6;
  do
  {
    v7 = (unsigned __int64)v4;
    v8 = *v4++;
    if ( !isxdigit(v8) )
    {
      if ( v14 )
      {
        sub_7B0EB0(v7, (__int64)dword_4F07508);
        sub_6851C0(0x3C5u, dword_4F07508);
      }
      goto LABEL_17;
    }
    v9 = 48;
    if ( (unsigned int)(v8 - 48) > 9 )
      v9 = islower(v8) == 0 ? 55 : 87;
    v6 = (16 * v6) | ((char)v8 - v9);
  }
  while ( (unsigned __int8 *)v15 != v4 );
  if ( v14 )
  {
    if ( dword_4F077C4 != 2 )
    {
      if ( v6 == 36 && a2 )
      {
        if ( !unk_4D04748 )
          goto LABEL_13;
LABEL_40:
        v10 = 967;
LABEL_39:
        v7 = (unsigned __int64)v4;
        sub_7B0EB0(v13, (__int64)dword_4F07508);
        sub_684AC0(byte_4F07472[0], v10);
        goto LABEL_17;
      }
      if ( v6 <= 0x9F && v6 != 36 )
      {
        if ( (v6 & 0xFFFFFFFFFFFFFFDFLL) != 0x40 )
          goto LABEL_40;
        goto LABEL_28;
      }
      v10 = 1661;
      if ( v6 - 55296 <= 0x7FF )
        goto LABEL_39;
      if ( v6 > 0x10FFFF )
      {
LABEL_38:
        v10 = 2215;
        goto LABEL_39;
      }
LABEL_28:
      if ( !a2 )
      {
        v7 = (unsigned __int64)v4;
        goto LABEL_17;
      }
LABEL_13:
      v7 = (unsigned __int64)v4;
      v10 = sub_7AC070(v6, a3);
      if ( !v10 )
        goto LABEL_17;
      v13 = *a1;
      goto LABEL_39;
    }
    if ( unk_4F07778 > 201102 || unk_4D043A8 | dword_4F07774 )
    {
      if ( v6 - 55296 <= 0x7FF )
      {
        v10 = 1661;
        goto LABEL_39;
      }
      if ( v6 > 0x10FFFF )
        goto LABEL_38;
      v7 = (unsigned __int64)v4;
      if ( !a2 )
        goto LABEL_17;
      if ( v6 > 0xFF || (unsigned int)sub_7B3970(v6) )
        goto LABEL_13;
    }
    else if ( v6 > 0xFF || (unsigned int)sub_7B3970(v6) )
    {
      if ( v6 - 127 <= 0x20 || v6 <= 0x1F )
      {
        v10 = 966;
        v13 = *a1;
        goto LABEL_39;
      }
      v7 = (unsigned __int64)v4;
      if ( !a2 )
        goto LABEL_17;
      goto LABEL_13;
    }
    v10 = 967;
    v13 = *a1;
    goto LABEL_39;
  }
  v7 = v15;
LABEL_17:
  *a1 = v7;
  return v6;
}

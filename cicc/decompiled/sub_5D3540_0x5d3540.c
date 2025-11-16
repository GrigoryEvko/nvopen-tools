// Function: sub_5D3540
// Address: 0x5d3540
//
void __fastcall sub_5D3540(__int64 a1)
{
  __int64 i; // rbx
  bool v2; // zf
  unsigned __int64 v3; // rax

  for ( i = *(_QWORD *)(a1 + 120); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(a1 + 144) & 4) != 0 )
  {
    if ( !qword_4CF7C90 )
    {
      v3 = *(_QWORD *)(a1 + 128);
LABEL_11:
      qword_4CF7C90 = i;
      qword_4CF7C98 = v3;
      return;
    }
    v2 = (unsigned int)sub_8E38C0(i) == 0;
    v3 = *(_QWORD *)(a1 + 128);
    if ( v2 || v3 >= qword_4CF7C98 + *(_QWORD *)(i + 128) )
      goto LABEL_11;
  }
  else
  {
    qword_4CF7C90 = 0;
  }
}

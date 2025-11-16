// Function: sub_6455C0
// Address: 0x6455c0
//
__int64 __fastcall sub_6455C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // r12
  char v6; // al
  __int64 result; // rax
  __int64 v8; // rdi

  if ( qword_4D0495C )
  {
    v3 = *(_QWORD *)(a1 + 288);
    if ( (unsigned int)sub_6454D0(v3, a2) )
      goto LABEL_19;
  }
  v4 = *(_QWORD *)(a1 + 288);
  v5 = v4;
  if ( *(_BYTE *)(v4 + 140) != 12 )
  {
LABEL_12:
    if ( dword_4F077C4 != 2 )
      goto LABEL_8;
    v4 = *(_QWORD *)(a1 + 288);
    if ( (*(_BYTE *)(v4 + 140) & 0xFB) != 8 )
      goto LABEL_8;
    goto LABEL_14;
  }
  do
  {
    v5 = *(_QWORD *)(v5 + 160);
    v6 = *(_BYTE *)(v5 + 140);
  }
  while ( v6 == 12 );
  if ( *(_QWORD *)(v4 + 8) && v6 == 7 )
  {
    if ( (unsigned int)sub_8D3150(v5) )
      sub_6851C0(989, a2);
    goto LABEL_12;
  }
  if ( dword_4F077C4 != 2 )
    goto LABEL_8;
LABEL_14:
  if ( (sub_8D4C10(v4, 0) & 2) != 0 )
  {
    v8 = 4;
    if ( dword_4F077C4 == 2 )
      v8 = (unsigned int)(unk_4F07778 > 202001) + 4;
    sub_684AA0(v8, 3013, a2);
  }
LABEL_8:
  sub_645520((__int64 *)(a1 + 288));
  result = sub_8D2600(v5);
  if ( (_DWORD)result )
  {
    v3 = 526;
    sub_6851C0(526, a2);
LABEL_19:
    result = sub_72C930(v3);
    *(_QWORD *)(a1 + 288) = result;
  }
  return result;
}

// Function: sub_65B9A0
// Address: 0x65b9a0
//
_BOOL8 __fastcall sub_65B9A0(int *a1)
{
  int v1; // eax
  _BOOL8 v2; // r12
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 v6; // rcx
  _BYTE v7[80]; // [rsp+0h] [rbp-50h] BYREF

  if ( !unk_4D0440C )
    goto LABEL_4;
  if ( dword_4F077C4 == 2 )
  {
    if ( (word_4F06418[0] != 1 || (unk_4D04A11 & 2) == 0) && !(unsigned int)sub_7C0F00(0, 0) )
      goto LABEL_4;
  }
  else if ( word_4F06418[0] != 1 )
  {
LABEL_4:
    v1 = 0;
    LODWORD(v2) = 0;
    goto LABEL_5;
  }
  v4 = sub_7BE840(0, 0);
  if ( v4 == 56 )
  {
    v1 = 0;
    LODWORD(v2) = 1;
    goto LABEL_5;
  }
  if ( v4 == 25 )
  {
    if ( !dword_4D043F8 )
      goto LABEL_4;
  }
  else if ( v4 != 142 || !dword_4D043E0 )
  {
    goto LABEL_4;
  }
  LODWORD(v2) = 1;
  v1 = unk_4D0418C;
  if ( unk_4D0418C )
  {
    sub_7ADF70(v7, 0);
    sub_7AE360(v7);
    sub_7B8B50(v7, 0, v5, v6);
    ++*(_BYTE *)(qword_4F061C8 + 83LL);
    sub_672540(v7);
    v2 = word_4F06418[0] == 56;
    --*(_BYTE *)(qword_4F061C8 + 83LL);
    sub_7BC000(v7);
    v1 = 1;
  }
LABEL_5:
  *a1 = v1;
  return v2;
}

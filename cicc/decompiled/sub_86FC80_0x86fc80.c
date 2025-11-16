// Function: sub_86FC80
// Address: 0x86fc80
//
__int64 sub_86FC80()
{
  __int64 v0; // rax
  __int64 v1; // rdx

  if ( unk_4D03B90 <= 0 )
    return 0;
  v0 = qword_4D03B98 + 176LL * unk_4D03B90;
  v1 = v0 - 176 - 176LL * (unsigned int)(unk_4D03B90 - 1);
  while ( (*(_BYTE *)(v0 + 4) & 0x20) == 0 )
  {
    v0 -= 176;
    if ( v0 == v1 )
      return 0;
  }
  return 1;
}

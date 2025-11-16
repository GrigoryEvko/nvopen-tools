// Function: sub_72C390
// Address: 0x72c390
//
__int64 sub_72C390()
{
  __int64 v1; // r12
  char v2; // al
  _QWORD *v3; // rdx
  __int64 v4; // rax

  if ( qword_4F07B68 )
    return qword_4F07B68;
  qword_4F07B68 = (__int64)sub_7259C0(2);
  v1 = qword_4F07B68;
  if ( dword_4F077C4 == 2 )
    v2 = unk_4F06B39;
  else
    v2 = unk_4F06B38;
  *(_BYTE *)(qword_4F07B68 + 162) |= 4u;
  *(_BYTE *)(v1 + 160) = v2;
  sub_8D6090(v1);
  if ( !*(_QWORD *)(v1 - 16) )
  {
    v3 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
    v4 = v3[13];
    if ( v1 != v4 )
    {
      if ( v4 )
        *(_QWORD *)(v4 - 16) = v1;
      else
        v3[12] = v1;
      v3[13] = v1;
    }
  }
  sub_8CC670(v1);
  return v1;
}

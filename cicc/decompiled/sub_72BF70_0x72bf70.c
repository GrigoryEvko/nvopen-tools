// Function: sub_72BF70
// Address: 0x72bf70
//
__int64 sub_72BF70()
{
  _QWORD *v1; // r12
  _QWORD *v2; // rdx
  _QWORD *v3; // rax

  if ( qword_4F07B88 )
    return qword_4F07B88;
  v1 = sub_7259C0(2);
  qword_4F07B88 = (__int64)v1;
  *((_BYTE *)v1 + 161) |= 0x40u;
  *((_BYTE *)v1 + 160) = byte_4F06B90[0];
  sub_8D6090(v1);
  if ( !*(v1 - 2) )
  {
    v2 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
    v3 = (_QWORD *)v2[13];
    if ( v1 != v3 )
    {
      if ( v3 )
        *(v3 - 2) = v1;
      else
        v2[12] = v1;
      v2[13] = v1;
    }
  }
  sub_8CC670(v1);
  return (__int64)v1;
}

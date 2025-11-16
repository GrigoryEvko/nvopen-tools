// Function: sub_72C1B0
// Address: 0x72c1b0
//
__int64 sub_72C1B0()
{
  _QWORD *v1; // r12
  char v2; // al
  _QWORD *v3; // rdx
  _QWORD *v4; // rax

  if ( qword_4F07B70 )
    return qword_4F07B70;
  v1 = sub_7259C0(2);
  qword_4F07B70 = (__int64)v1;
  v2 = unk_4F06B70;
  *((_BYTE *)v1 + 162) |= 2u;
  *((_BYTE *)v1 + 160) = v2;
  sub_8D6090(v1);
  if ( !*(v1 - 2) )
  {
    v3 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
    v4 = (_QWORD *)v3[13];
    if ( v1 != v4 )
    {
      if ( v4 )
        *(v4 - 2) = v1;
      else
        v3[12] = v1;
      v3[13] = v1;
    }
  }
  sub_8CC670(v1);
  return (__int64)v1;
}

// Function: sub_72BCF0
// Address: 0x72bcf0
//
_QWORD *__fastcall sub_72BCF0(unsigned __int8 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r12
  _QWORD *v4; // rdx
  _QWORD *v5; // rax

  if ( qword_4F07E40[a1] )
    return (_QWORD *)qword_4F07E40[a1];
  v2 = sub_7259C0(2);
  *((_BYTE *)v2 + 161) |= 1u;
  v3 = v2;
  *((_BYTE *)v2 + 160) = a1;
  qword_4F07E40[a1] = v2;
  sub_8D6090(v2);
  if ( !*(v3 - 2) )
  {
    v4 = dword_4F07588 ? *(_QWORD **)(unk_4D03FF0 + 376LL) : &unk_4F06D00;
    v5 = (_QWORD *)v4[13];
    if ( v3 != v5 )
    {
      if ( v5 )
        *(v5 - 2) = v3;
      else
        v4[12] = v3;
      v4[13] = v3;
    }
  }
  sub_8CC670(v3);
  return v3;
}

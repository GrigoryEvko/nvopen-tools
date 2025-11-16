// Function: sub_31DCB40
// Address: 0x31dcb40
//
void __fastcall sub_31DCB40(__int64 a1, signed __int64 a2, __int64 a3)
{
  _BYTE *v3; // rax
  __int64 v4; // rax

  if ( a2 > 0 )
  {
    v3 = *(_BYTE **)(a3 + 32);
    if ( (unsigned __int64)v3 >= *(_QWORD *)(a3 + 24) )
    {
      v4 = sub_CB5D20(a3, 43);
      sub_CB59F0(v4, a2);
      return;
    }
    *(_QWORD *)(a3 + 32) = v3 + 1;
    *v3 = 43;
LABEL_6:
    sub_CB59F0(a3, a2);
    return;
  }
  if ( a2 )
    goto LABEL_6;
}

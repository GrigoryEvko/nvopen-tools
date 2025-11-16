// Function: sub_897810
// Address: 0x897810
//
_QWORD *__fastcall sub_897810(unsigned __int8 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // ebx
  _QWORD *result; // rax

  v4 = a3;
  if ( (_DWORD)a4 || (unsigned int)sub_866580() )
  {
    if ( !v4 )
    {
      result = sub_87EBB0(a1, *(_QWORD *)a2, (_QWORD *)(a2 + 8));
      goto LABEL_4;
    }
  }
  else if ( !v4 )
  {
    result = sub_885AD0(a1, a2, dword_4F04C5C, 0);
    *((_BYTE *)result + 83) |= 0x40u;
    goto LABEL_5;
  }
  if ( !a2 || (*(_BYTE *)(a2 + 17) & 0x20) != 0 )
  {
    result = sub_87F7E0(a1, &dword_4F063F8, a3, a4);
    goto LABEL_5;
  }
  result = sub_87EF90(a1, a2);
LABEL_4:
  *((_DWORD *)result + 10) = *(_DWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C5C);
LABEL_5:
  *((_BYTE *)result + 81) |= 0x40u;
  *((_DWORD *)result + 14) = dword_4F06650[0];
  return result;
}

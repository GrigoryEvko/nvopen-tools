// Function: sub_6A9E80
// Address: 0x6a9e80
//
__int64 __fastcall sub_6A9E80(_QWORD *a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rdx
  __int64 result; // rax

  if ( a1 )
  {
    v2 = *(unsigned __int8 *)(*a1 + 56LL);
    v3 = sub_68AFD0(*(_BYTE *)(*a1 + 56LL));
    if ( (_BYTE)v2 != 74 )
      goto LABEL_3;
  }
  else
  {
    if ( word_4F06418[0] != 300 )
    {
      v2 = 75;
      v3 = sub_68AFD0(0x4Bu);
LABEL_3:
      sub_6A9320(a1, v2, v3, 4, 0, 0, a2);
      result = dword_4D044B0;
      if ( dword_4D044B0 )
        return result;
      return sub_6E6840(a2);
    }
    v3 = sub_68AFD0(0x4Au);
  }
  sub_6A9320(a1, 74, v3, 1, 4u, 0, a2);
  result = dword_4D044B0;
  if ( !dword_4D044B0 )
    return sub_6E6840(a2);
  return result;
}

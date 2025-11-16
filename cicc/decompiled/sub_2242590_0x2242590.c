// Function: sub_2242590
// Address: 0x2242590
//
__int64 __fastcall sub_2242590(wchar_t *s2, __int64 a2)
{
  __int64 v2; // rsi
  size_t v3; // r13
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // r14

  v2 = a2 - (_QWORD)s2;
  v3 = v2 >> 2;
  v4 = sub_2216040(v2 >> 2, 0);
  v5 = v4;
  v6 = v4 + 24;
  if ( v2 >> 2 == 1 )
  {
    *(_DWORD *)(v4 + 24) = *s2;
  }
  else if ( v3 )
  {
    wmemcpy((wchar_t *)(v4 + 24), s2, v3);
    if ( (_UNKNOWN *)v5 == &unk_4FD67E0 )
      return v6;
    goto LABEL_6;
  }
  if ( (_UNKNOWN *)v4 == &unk_4FD67E0 )
    return v6;
LABEL_6:
  *(_DWORD *)(v5 + 16) = 0;
  *(_QWORD *)v5 = v3;
  *(_DWORD *)(v5 + v2 + 24) = 0;
  return v6;
}

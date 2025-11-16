// Function: sub_657FD0
// Address: 0x657fd0
//
__int64 __fastcall sub_657FD0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  int v5; // eax

  result = *(_QWORD *)(a1 + 8);
  if ( (result & 2) == 0 )
    goto LABEL_5;
  if ( !a3 )
  {
    v4 = a1 + 88;
LABEL_4:
    sub_684AA0(dword_4F077C0 == 0 ? 8 : 5, 325, v4);
    result = *(_QWORD *)(a1 + 8);
    goto LABEL_5;
  }
  if ( !dword_4D04820 )
  {
    v4 = a1 + 88;
    if ( !dword_4F077BC )
      goto LABEL_4;
    v5 = sub_657F30((unsigned int *)(a1 + 88));
    v4 = a1 + 88;
    if ( !v5 )
      goto LABEL_4;
    result = *(_QWORD *)(a1 + 8);
  }
LABEL_5:
  if ( (result & 1) == 0 )
    return sub_64EBC0(a1, 0, a2);
  return result;
}

// Function: sub_2215DF0
// Address: 0x2215df0
//
__int64 __fastcall sub_2215DF0(__int64 *a1, char a2)
{
  __int64 v2; // r13
  __int64 result; // rax

  v2 = *(_QWORD *)(*a1 - 24);
  if ( (unsigned __int64)(v2 + 1) > *(_QWORD *)(*a1 - 16) || *(int *)(*a1 - 8) > 0 )
    sub_2215AB0(a1, v2 + 1);
  *(_BYTE *)(*a1 + *(_QWORD *)(*a1 - 24)) = a2;
  result = *a1;
  if ( (_UNKNOWN *)(*a1 - 24) != &unk_4FD67C0 )
  {
    *(_DWORD *)(result - 8) = 0;
    *(_QWORD *)(result - 24) = v2 + 1;
    *(_BYTE *)(result + v2 + 1) = 0;
  }
  return result;
}

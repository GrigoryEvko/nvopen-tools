// Function: sub_C3CF20
// Address: 0xc3cf20
//
__int64 __fastcall sub_C3CF20(__int64 a1, unsigned __int8 a2)
{
  _DWORD **v3; // rdi
  _DWORD *v4; // rbx
  void **v5; // rdi

  v3 = *(_DWORD ***)(a1 + 8);
  v4 = sub_C33340();
  if ( *v3 == v4 )
    sub_C3CF20(v3, a2);
  else
    sub_C36EF0(v3, a2);
  v5 = (void **)(*(_QWORD *)(a1 + 8) + 24LL);
  if ( *v5 == v4 )
    return sub_C3CEB0(v5, 0);
  else
    return sub_C37310((__int64)v5, 0);
}

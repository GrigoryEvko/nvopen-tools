// Function: sub_3909D40
// Address: 0x3909d40
//
__int64 __fastcall sub_3909D40(unsigned int *a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rax
  bool v6; // cc
  _QWORD *v7; // rax

  if ( *(_DWORD *)sub_3909460((__int64)a1) != 4 )
    return sub_3909CF0(a1, a3, 0, 0);
  v5 = sub_3909460((__int64)a1);
  v6 = *(_DWORD *)(v5 + 32) <= 0x40u;
  v7 = *(_QWORD **)(v5 + 24);
  if ( !v6 )
    v7 = (_QWORD *)*v7;
  *a2 = v7;
  (*(void (__fastcall **)(unsigned int *))(*(_QWORD *)a1 + 136LL))(a1);
  return 0;
}

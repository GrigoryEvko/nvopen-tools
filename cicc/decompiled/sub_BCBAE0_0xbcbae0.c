// Function: sub_BCBAE0
// Address: 0xbcbae0
//
__int64 __fastcall sub_BCBAE0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int8 *v3; // rax
  bool v4; // cc
  _QWORD *v5; // rax

  v3 = sub_AD8340(a2, (__int64)a2, a3);
  v4 = *((_DWORD *)v3 + 2) <= 0x40u;
  v5 = *(_QWORD **)v3;
  if ( !v4 )
    v5 = (_QWORD *)*v5;
  return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * (unsigned int)v5);
}

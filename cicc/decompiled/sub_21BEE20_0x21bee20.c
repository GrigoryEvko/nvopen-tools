// Function: sub_21BEE20
// Address: 0x21bee20
//
__int64 __fastcall sub_21BEE20(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  _QWORD *v3; // rax
  unsigned int v4; // r8d

  v2 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 88LL);
  v3 = *(_QWORD **)(v2 + 24);
  if ( *(_DWORD *)(v2 + 32) > 0x40u )
    v3 = (_QWORD *)*v3;
  if ( (_DWORD)v3 == 3760 )
  {
    sub_21BED00(a1, a2);
    return 1;
  }
  else
  {
    v4 = 0;
    if ( (_DWORD)v3 == 5233 )
    {
      sub_21BED50(a1, a2);
      return 1;
    }
    return v4;
  }
}

// Function: sub_5E39A0
// Address: 0x5e39a0
//
void __fastcall sub_5E39A0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  unsigned int v7; // r12d
  __int64 i; // rbx
  _QWORD *j; // rbx

  v7 = a3;
  for ( i = *(_QWORD *)(a1 + 104); i; i = *(_QWORD *)(i + 112) )
  {
    *a4 = 1;
    if ( v7 == 1 )
    {
      sub_5E3920(i, a2, *(_DWORD *)(a1 + 24));
      *(_BYTE *)(i + 141) &= ~0x80u;
    }
    sub_5DB980((FILE *)i, v7, a3, (__int64)a4, a5, a6);
  }
  for ( j = *(_QWORD **)(a1 + 160); j; j = (_QWORD *)*j )
    sub_5E39A0(j, a2, v7, a4);
}

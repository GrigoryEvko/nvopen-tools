// Function: sub_39A3770
// Address: 0x39a3770
//
void __fastcall sub_39A3770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx

  v3 = a3;
  if ( *(_BYTE *)a3 != 15 )
    v3 = *(_QWORD *)(a3 - 8LL * *(unsigned int *)(a3 + 8));
  sub_39A36D0(a1, a2, *(_DWORD *)(a3 + 24), v3);
}

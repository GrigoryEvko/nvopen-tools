// Function: sub_2252810
// Address: 0x2252810
//
__int64 __fastcall sub_2252810(_QWORD *a1)
{
  __int64 v2; // rax
  _QWORD *v3; // rcx
  _QWORD *v4; // r8
  int v6; // edi
  int v7; // edx

  v2 = sub_22529C0();
  v3 = a1 - 10;
  v4 = *(_QWORD **)v2;
  if ( (unsigned __int64)(*a1 - 0x474E5543432B2B00LL) <= 1 )
  {
    v6 = *((_DWORD *)a1 - 10);
    v7 = v6 + 1;
    if ( v6 < 0 )
      v7 = 1 - v6;
    *((_DWORD *)a1 - 10) = v7;
    --*(_DWORD *)(v2 + 8);
    if ( v4 != v3 )
    {
      *(a1 - 6) = v4;
      *(_QWORD *)v2 = v3;
    }
    return *(a1 - 1);
  }
  else
  {
    if ( v4 )
      sub_2207530();
    *(_QWORD *)v2 = v3;
    return 0;
  }
}

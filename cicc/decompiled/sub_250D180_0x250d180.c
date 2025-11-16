// Function: sub_250D180
// Address: 0x250d180
//
__int64 __fastcall sub_250D180(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdx
  _BYTE *v3; // rax

  v2 = *a1 & 3;
  if ( v2 == 3 )
    return *(_QWORD *)(sub_250D070(a1) + 8);
  if ( v2 == 2 )
    return *(_QWORD *)(sub_250D070(a1) + 8);
  v3 = (_BYTE *)(*a1 & 0xFFFFFFFFFFFFFFFCLL);
  if ( !v3 || *v3 || v2 != 1 )
    return *(_QWORD *)(sub_250D070(a1) + 8);
  else
    return **(_QWORD **)(*((_QWORD *)sub_250CBE0(a1, a2) + 3) + 16LL);
}

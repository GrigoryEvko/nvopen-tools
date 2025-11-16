// Function: sub_C8CF80
// Address: 0xc8cf80
//
__int64 __fastcall sub_C8CF80(__int64 a1, void *a2, int a3, __int64 a4, __int64 a5)
{
  if ( *(_BYTE *)(a1 + 28) )
    return sub_C8CEE0(a1, a2, a3, a4, a5);
  _libc_free(*(_QWORD *)(a1 + 8), a2);
  return sub_C8CEE0(a1, a2, a3, a4, a5);
}

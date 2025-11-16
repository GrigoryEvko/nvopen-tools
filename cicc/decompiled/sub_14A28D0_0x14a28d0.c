// Function: sub_14A28D0
// Address: 0x14a28d0
//
__int64 __fastcall sub_14A28D0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 96LL);
  if ( (char *)v3 != (char *)sub_14A07E0 )
    return v3();
  *a3 = 0;
  return ((*(_DWORD *)(a2 + 20) & 0xFFFFFFFu) >> 1) - 1;
}

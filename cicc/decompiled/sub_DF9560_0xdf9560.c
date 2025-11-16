// Function: sub_DF9560
// Address: 0xdf9560
//
__int64 __fastcall sub_DF9560(__int64 a1, __int64 a2, _DWORD *a3)
{
  __int64 (*v3)(void); // rax

  v3 = *(__int64 (**)(void))(**(_QWORD **)a1 + 112LL);
  if ( (char *)v3 != (char *)sub_DF5C00 )
    return v3();
  *a3 = 0;
  return ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
}

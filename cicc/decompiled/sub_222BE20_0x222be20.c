// Function: sub_222BE20
// Address: 0x222be20
//
__int64 __fastcall sub_222BE20(__int64 a1, __int64 a2)
{
  if ( (*(unsigned __int8 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 200) + 48LL))(*(_QWORD *)(a1 + 200)) )
    return (unsigned int)*(_QWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24);
  else
    return *(_DWORD *)(a1 + 208)
         + (*(unsigned int (__fastcall **)(_QWORD, __int64, _QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 200) + 56LL))(
             *(_QWORD *)(a1 + 200),
             a2,
             *(_QWORD *)(a1 + 208),
             *(_QWORD *)(a1 + 224),
             *(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8))
         - *(_DWORD *)(a1 + 232);
}

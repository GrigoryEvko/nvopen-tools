// Function: sub_2E8A3A0
// Address: 0x2e8a3a0
//
__int64 __fastcall sub_2E8A3A0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v7; // rdx
  _DWORD *v8; // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 (__fastcall *v12)(__int64, __int64); // rax

  v7 = sub_2E8A250(a1, a2, a4, a5);
  v8 = (_DWORD *)(*(_QWORD *)(a1 + 32) + 40LL * a2);
  v9 = (*v8 >> 8) & 0xFFF;
  if ( ((*v8 >> 8) & 0xFFF) == 0 )
  {
    if ( v7 )
      return sub_2FF6970(a5, a3, v7, v9);
    return a3;
  }
  v10 = *a5;
  if ( v7 )
    return (*(__int64 (__fastcall **)(__int64 *, __int64))(v10 + 256))(a5, a3);
  v12 = *(__int64 (__fastcall **)(__int64, __int64))(v10 + 272);
  if ( v12 == sub_2E85430 )
    return a3;
  return ((__int64 (__fastcall *)(__int64 *, __int64, _QWORD))v12)(a5, a3, (unsigned int)v9);
}

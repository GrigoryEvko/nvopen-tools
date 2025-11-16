// Function: sub_1E9C310
// Address: 0x1e9c310
//
bool __fastcall sub_1E9C310(_QWORD *a1, unsigned int a2)
{
  __int64 (*v2)(); // rax
  __int64 v3; // rcx
  bool result; // al

  v2 = *(__int64 (**)())(**(_QWORD **)(*a1 + 16LL) + 112LL);
  if ( v2 == sub_1D00B10 )
    BUG();
  v3 = *(_QWORD *)(v2() + 232);
  result = 1;
  if ( *(_BYTE *)(v3 + 8LL * a2 + 4) )
    return (*(_QWORD *)(a1[38] + 8LL * (a2 >> 6)) & (1LL << a2)) != 0;
  return result;
}

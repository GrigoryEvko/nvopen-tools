// Function: sub_1EE5C30
// Address: 0x1ee5c30
//
__int64 __fastcall sub_1EE5C30(__int64 *a1, _QWORD *a2, int a3)
{
  __int64 v3; // r14
  __int64 (*v5)(void); // rax
  __int64 v6; // rax
  _DWORD *v7; // rbx
  int v8; // ecx
  __int64 result; // rax
  __int64 v10; // rdx
  unsigned __int64 v11; // r13

  v3 = 0;
  v5 = *(__int64 (**)(void))(**(_QWORD **)(*a2 + 16LL) + 112LL);
  if ( v5 != sub_1D00B10 )
    v3 = v5();
  v6 = *(_QWORD *)v3;
  if ( a3 < 0 )
  {
    v11 = *(_QWORD *)(a2[3] + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v7 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(v6 + 224))(v3, v11);
    v8 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v3 + 184LL))(v3, v11);
  }
  else
  {
    v7 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(v6 + 232))(v3, (unsigned int)a3);
    v8 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 192LL))(v3, (unsigned int)a3);
  }
  result = 0;
  if ( *v7 == -1 )
    v7 = 0;
  do
  {
    if ( !v7 )
      break;
    v10 = (unsigned int)*v7;
    result = *a1;
    ++v7;
    *(_DWORD *)(*a1 + 4 * v10) += v8;
  }
  while ( *v7 != -1 );
  return result;
}

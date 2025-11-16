// Function: sub_2F74DB0
// Address: 0x2f74db0
//
__int64 __fastcall sub_2F74DB0(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v10; // r14
  __int64 v11; // rax
  __int64 v12; // rdi
  _DWORD *v13; // r14
  int v14; // esi
  __int64 result; // rax
  __int64 v16; // rdx
  __int64 v17; // rdi
  _DWORD *v18; // rcx
  unsigned __int64 v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  __int128 v21; // [rsp+10h] [rbp-40h]

  v10 = *(_QWORD **)(a1 + 24);
  *((_QWORD *)&v21 + 1) = a5;
  *(_QWORD *)&v21 = a6;
  v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*v10 + 16LL) + 200LL))(*(_QWORD *)(*v10 + 16LL));
  v12 = v11;
  if ( (a2 & 0x80000000) != 0 )
  {
    v20 = v11;
    v19 = *(_QWORD *)(v10[7] + 16LL * (a2 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v13 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 416LL))(v11);
    v14 = *(_DWORD *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v20 + 376LL))(v20, v19);
  }
  else
  {
    v13 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v11 + 424LL))(v11, a2);
    v14 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v12 + 384LL))(v12, a2);
  }
  result = 0;
  if ( *v13 == -1 )
    v13 = 0;
  if ( *(_BYTE *)(a1 + 58)
    && (result = *(_QWORD *)(**(_QWORD **)(a1 + 8) + 432LL), (__int64 (*)())result != sub_2F73F20)
    && (result = ((__int64 (*)(void))result)(), (_BYTE)result) )
  {
    result = v21 & ~a4 | ~a3 & *((_QWORD *)&v21 + 1);
    if ( !result )
      return result;
    result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 8) + 440LL))(
               *(_QWORD *)(a1 + 8),
               *(_QWORD *)(a1 + 24),
               a2);
    v14 = result;
  }
  else
  {
    if ( a3 | a4 )
      return result;
    result = v21 | *((_QWORD *)&v21 + 1);
    if ( v21 == 0 )
      return result;
  }
  do
  {
    if ( !v13 )
      break;
    *(_DWORD *)(*(_QWORD *)(a1 + 72) + 4LL * (unsigned int)*v13) += v14;
    v16 = (unsigned int)*v13;
    v17 = *(_QWORD *)(a1 + 72);
    v18 = (_DWORD *)(**(_QWORD **)(a1 + 48) + 4 * v16);
    result = (unsigned int)*v18;
    if ( *(_DWORD *)(v17 + 4 * v16) >= (unsigned int)result )
      result = *(unsigned int *)(v17 + 4 * v16);
    ++v13;
    *v18 = result;
  }
  while ( *v13 != -1 );
  return result;
}

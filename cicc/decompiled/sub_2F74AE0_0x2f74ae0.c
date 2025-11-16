// Function: sub_2F74AE0
// Address: 0x2f74ae0
//
__int64 __fastcall sub_2F74AE0(
        __int64 *a1,
        _QWORD *a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rsi
  _DWORD *v15; // r15
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 result; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int128 v22; // [rsp+10h] [rbp-50h]
  __int128 v23; // [rsp+20h] [rbp-40h]

  *((_QWORD *)&v23 + 1) = a4;
  *(_QWORD *)&v23 = a5;
  *((_QWORD *)&v22 + 1) = a7;
  *(_QWORD *)&v22 = a8;
  v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*a2 + 16LL) + 200LL))(*(_QWORD *)(*a2 + 16LL));
  v13 = v12;
  if ( (a3 & 0x80000000) != 0 )
  {
    v21 = v12;
    v14 = *(_QWORD *)(a2[7] + 16LL * (a3 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
    v15 = (_DWORD *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v12 + 416LL))(v12);
    v17 = *(unsigned int *)(*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)v21 + 376LL))(v21, v14);
  }
  else
  {
    v14 = a3;
    v15 = (_DWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v12 + 424LL))(v12, a3);
    v17 = (*(unsigned int (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v13 + 384LL))(v13, a3);
  }
  if ( *v15 == -1 )
    v15 = 0;
  if ( a6 )
  {
    result = ~(_QWORD)v23 & a8 | a7 & ~*((_QWORD *)&v23 + 1);
    if ( (~v23 & v22) == 0 )
      return result;
    v19 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, __int64, __int64))(**(_QWORD **)(*a2 + 16LL) + 200LL))(
            *(_QWORD *)(*a2 + 16LL),
            v14,
            v16,
            v17);
    result = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD, __int64, __int64))(*(_QWORD *)v19 + 440LL))(
               v19,
               a2,
               a3,
               a7 & ~*((_QWORD *)&v23 + 1),
               ~(_QWORD)v23 & a8);
    LODWORD(v17) = result;
  }
  else
  {
    result = v23 | *((_QWORD *)&v23 + 1);
    if ( v23 != 0 )
      return result;
    result = a8 | a7;
    if ( v22 == 0 )
      return result;
  }
  do
  {
    if ( !v15 )
      break;
    v20 = (unsigned int)*v15;
    result = *a1;
    ++v15;
    *(_DWORD *)(*a1 + 4 * v20) += v17;
  }
  while ( *v15 != -1 );
  return result;
}

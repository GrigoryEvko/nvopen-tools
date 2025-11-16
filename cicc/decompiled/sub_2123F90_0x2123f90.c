// Function: sub_2123F90
// Address: 0x2123f90
//
__int64 *__fastcall sub_2123F90(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v3; // r8
  __int64 v4; // rax

  v3 = sub_2120330(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v4 = *(_QWORD *)(a2 + 32);
  if ( *(_QWORD *)v4 == v3 && *(_DWORD *)(v4 + 8) == (_DWORD)v2 )
    return 0;
  else
    return sub_1D2DE40(*(_QWORD **)(a1 + 8), (__int64 *)a2, v3, v2);
}

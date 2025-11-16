// Function: sub_2123FF0
// Address: 0x2123ff0
//
__int64 *__fastcall sub_2123FF0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // rdx
  unsigned __int64 v6; // r8
  __int64 v7; // rax

  v2 = sub_2120330(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v4 = v3;
  v6 = sub_2120330(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL));
  v7 = *(_QWORD *)(a2 + 32);
  if ( *(_QWORD *)v7 == v2
    && *(_DWORD *)(v7 + 8) == (_DWORD)v4
    && *(_QWORD *)(v7 + 40) == v6
    && *(_DWORD *)(v7 + 48) == (_DWORD)v5 )
  {
    return 0;
  }
  else
  {
    return sub_1D2DF70(*(_QWORD **)(a1 + 8), (__int64 *)a2, v2, v4, v6, v5);
  }
}

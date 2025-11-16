// Function: sub_B86040
// Address: 0xb86040
//
__int64 __fastcall sub_B86040(__int64 a1, __int64 a2)
{
  unsigned int v3; // r13d
  __int64 *v4; // rbx
  __int64 *i; // r15
  __int64 v6; // rdi
  __int64 (*v7)(); // rax
  unsigned int j; // ebx
  __int64 v9; // rax

  sub_B85E70(a1 + 568);
  v3 = 0;
  sub_B80B80(a1 + 568);
  v4 = *(__int64 **)(a1 + 824);
  for ( i = &v4[*(unsigned int *)(a1 + 832)];
        i != v4;
        v3 |= ((__int64 (__fastcall *)(__int64, __int64, __int64 (*)()))v7)(v6, a2, sub_97DD00) )
  {
    while ( 1 )
    {
      v6 = *v4;
      v7 = *(__int64 (**)())(*(_QWORD *)*v4 + 24LL);
      if ( v7 != sub_97DD00 )
        break;
      if ( i == ++v4 )
        goto LABEL_6;
    }
    ++v4;
  }
LABEL_6:
  for ( j = 0;
        j < *(_DWORD *)(a1 + 608);
        v3 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(v9 - 176) + 24LL))(v9 - 176, a2) )
  {
    v9 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + 8LL * j);
    if ( !v9 )
      BUG();
    ++j;
  }
  return v3;
}

// Function: sub_B7EDA0
// Address: 0xb7eda0
//
__int64 __fastcall sub_B7EDA0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // r12d
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 *v6; // rbx
  __int64 *i; // r14
  __int64 v8; // rdi
  __int64 (*v9)(); // rax

  v2 = *(_DWORD *)(a1 + 608) - 1;
  if ( v2 < 0 )
  {
    v3 = 0;
  }
  else
  {
    v3 = 0;
    v4 = 8LL * v2;
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 600) + v4);
      if ( !v5 )
        BUG();
      --v2;
      v4 -= 8;
      v3 |= (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)(v5 - 176) + 32LL))(v5 - 176, a2);
    }
    while ( v2 != -1 );
  }
  v6 = *(__int64 **)(a1 + 824);
  for ( i = &v6[*(unsigned int *)(a1 + 832)]; i != v6; v3 |= ((__int64 (__fastcall *)(__int64, __int64))v9)(v8, a2) )
  {
    while ( 1 )
    {
      v8 = *v6;
      v9 = *(__int64 (**)())(*(_QWORD *)*v6 + 32LL);
      if ( v9 != sub_97DD10 )
        break;
      if ( i == ++v6 )
        return v3;
    }
    ++v6;
  }
  return v3;
}

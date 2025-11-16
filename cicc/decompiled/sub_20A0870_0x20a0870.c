// Function: sub_20A0870
// Address: 0x20a0870
//
__int64 __fastcall sub_20A0870(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // r15
  __int64 v4; // rax
  __int64 v5; // rbx
  int v6; // r12d
  int v7; // eax
  __int64 v9; // [rsp+8h] [rbp-38h]

  v3 = a2 + 16;
  if ( *(_DWORD *)(a2 + 72) > a3 )
    v3 = *(_QWORD *)(a2 + 64) + 56LL * a3 + 8;
  v4 = *(unsigned int *)(v3 + 8);
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v6 = -1;
    v9 = 32 * v4;
    do
    {
      v7 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 1368LL))(
             a1,
             a2,
             *(_QWORD *)(*(_QWORD *)v3 + v5));
      if ( v6 < v7 )
        v6 = v7;
      v5 += 32;
    }
    while ( v9 != v5 );
  }
  else
  {
    return (unsigned int)-1;
  }
  return (unsigned int)v6;
}

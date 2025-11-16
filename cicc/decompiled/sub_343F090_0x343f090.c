// Function: sub_343F090
// Address: 0x343f090
//
__int64 __fastcall sub_343F090(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  _QWORD *v4; // rbx
  int v5; // r13d
  __int64 v6; // r12
  int v7; // eax

  v3 = a2 + 16;
  if ( *(_DWORD *)(a2 + 72) > a3 )
    v3 = *(_QWORD *)(a2 + 64) + 56LL * a3 + 8;
  v4 = *(_QWORD **)v3;
  v5 = -1;
  v6 = *(_QWORD *)v3 + 32LL * *(unsigned int *)(v3 + 8);
  if ( v6 != *(_QWORD *)v3 )
  {
    do
    {
      v7 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 2472LL))(a1, a2, *v4);
      if ( v5 < v7 )
        v5 = v7;
      v4 += 4;
    }
    while ( (_QWORD *)v6 != v4 );
  }
  return (unsigned int)v5;
}

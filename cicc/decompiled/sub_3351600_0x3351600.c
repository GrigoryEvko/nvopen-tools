// Function: sub_3351600
// Address: 0x3351600
//
__int64 __fastcall sub_3351600(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  _QWORD *v7; // r12
  _QWORD *i; // r13
  unsigned __int64 v9; // rbx
  __int64 *v10; // rdi
  unsigned int v11; // eax

  v6 = 0;
  v7 = *(_QWORD **)(a1 + 120);
  for ( i = &v7[2 * *(unsigned int *)(a1 + 128)]; i != v7; v7 += 2 )
  {
    if ( (*v7 & 6) == 0 )
    {
      v9 = *v7 & 0xFFFFFFFFFFFFFFF8LL;
      v10 = (__int64 *)v9;
      if ( (*(_BYTE *)(v9 + 254) & 2) == 0 )
      {
        sub_2F8F770(v9, a2, a3, a4, a5, a6);
        v10 = (__int64 *)(*v7 & 0xFFFFFFFFFFFFFFF8LL);
      }
      a3 = *v10;
      v11 = *(_DWORD *)(v9 + 244);
      if ( *v10 && *(_DWORD *)(a3 + 24) == 49 )
        v11 = sub_3351600() + 1;
      if ( v6 < v11 )
        v6 = v11;
    }
  }
  return v6;
}

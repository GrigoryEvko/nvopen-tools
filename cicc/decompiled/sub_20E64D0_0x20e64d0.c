// Function: sub_20E64D0
// Address: 0x20e64d0
//
__int64 __fastcall sub_20E64D0(__int64 a1, unsigned int *a2)
{
  __int64 v4; // rax
  unsigned int v5; // esi
  __int64 v6; // rdx
  __int64 v7; // r8
  __int64 v8; // r12
  unsigned int v9; // ecx
  __int64 v10; // rax
  _BOOL8 v12; // rdi

  v4 = sub_22077B0(48);
  v5 = *a2;
  v6 = *(_QWORD *)(a1 + 16);
  v7 = a1 + 8;
  v8 = v4;
  *(_DWORD *)(v4 + 32) = v5;
  *(_QWORD *)(v4 + 40) = *((_QWORD *)a2 + 1);
  if ( v6 )
  {
    while ( 1 )
    {
      v9 = *(_DWORD *)(v6 + 32);
      v10 = *(_QWORD *)(v6 + 24);
      if ( v9 > v5 )
        v10 = *(_QWORD *)(v6 + 16);
      if ( !v10 )
        break;
      v6 = v10;
    }
    v12 = 1;
    if ( v7 != v6 )
      v12 = v9 > v5;
  }
  else
  {
    v6 = a1 + 8;
    v12 = 1;
  }
  sub_220F040(v12, v8, v6, a1 + 8);
  ++*(_QWORD *)(a1 + 40);
  return v8;
}

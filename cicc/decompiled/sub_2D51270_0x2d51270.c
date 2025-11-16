// Function: sub_2D51270
// Address: 0x2d51270
//
__int64 __fastcall sub_2D51270(__int64 a1, __int64 a2, const void *a3, size_t a4)
{
  int v6; // eax
  int v7; // eax
  __int64 v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rax
  int v11; // eax
  int v12; // eax
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rsi

  v6 = sub_C92610();
  v7 = sub_C92860((__int64 *)(a2 + 120), a3, a4, v6);
  if ( v7 != -1 )
  {
    v8 = *(_QWORD *)(a2 + 120);
    v9 = (__int64 *)(v8 + 8LL * v7);
    if ( v9 != (__int64 *)(v8 + 8LL * *(unsigned int *)(a2 + 128)) )
    {
      v10 = *v9;
      a3 = *(const void **)(v10 + 8);
      a4 = *(_QWORD *)(v10 + 16);
    }
  }
  v11 = sub_C92610();
  v12 = sub_C92860((__int64 *)(a2 + 96), a3, a4, v11);
  v15 = a1 + 24;
  if ( v12 == -1
    || (v16 = *(_QWORD *)(a2 + 96),
        v17 = (__int64 *)(v16 + 8LL * v12),
        v18 = v16 + 8LL * *(unsigned int *)(a2 + 104),
        v17 == (__int64 *)v18) )
  {
    *(_BYTE *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 0x300000000LL;
    *(_QWORD *)(a1 + 8) = v15;
    return a1;
  }
  else
  {
    v19 = *v17;
    *(_BYTE *)a1 = 1;
    *(_QWORD *)(a1 + 8) = v15;
    *(_QWORD *)(a1 + 16) = 0x300000000LL;
    if ( *(_DWORD *)(v19 + 16) )
      sub_2D50040(a1 + 8, v19 + 8, v18, v15, v13, v14);
    return a1;
  }
}

// Function: sub_DEF1D0
// Address: 0xdef1d0
//
bool __fastcall sub_DEF1D0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  int v9; // eax
  __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rdi
  unsigned int v13; // esi
  __int64 v14; // rdx
  __int64 v15; // r9
  int v17; // edx
  int v18; // r10d

  v5 = sub_DEEF40(a1, a2);
  v9 = sub_DC1810(v5, *(_QWORD *)(a1 + 112), v6, v7, v8);
  v10 = *(unsigned int *)(a1 + 56);
  v11 = a3 & ~v9;
  if ( (_DWORD)v10 )
  {
    v12 = *(_QWORD *)(a1 + 40);
    v13 = (v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v14 = v12 + 48LL * (((_DWORD)v10 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)));
    v15 = *(_QWORD *)(v14 + 24);
    if ( v15 == a2 )
    {
LABEL_3:
      if ( v14 != v12 + 48 * v10 )
        v11 &= ~*(_DWORD *)(v14 + 40);
    }
    else
    {
      v17 = 1;
      while ( v15 != -4096 )
      {
        v18 = v17 + 1;
        v13 = (v10 - 1) & (v17 + v13);
        v14 = v12 + 48LL * v13;
        v15 = *(_QWORD *)(v14 + 24);
        if ( v15 == a2 )
          goto LABEL_3;
        v17 = v18;
      }
    }
  }
  return v11 == 0;
}

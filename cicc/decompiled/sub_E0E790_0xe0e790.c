// Function: sub_E0E790
// Address: 0xe0e790
//
__int64 __fastcall sub_E0E790(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // rbx
  __int64 *v8; // r13
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  _QWORD *v11; // rax
  __int64 **v12; // rdx
  __int64 v13; // rdx
  __int64 **v15; // rax

  v7 = ((_DWORD)a2 + 15) & 0xFFFFFFF0;
  v8 = *(__int64 **)(a1 + 4096);
  v9 = v7 + v8[1];
  if ( v9 > 0xFEF )
  {
    if ( v7 > 0xFF0 )
    {
      v10 = v7 + 16;
      v11 = (_QWORD *)malloc(v7 + 16, a2, a3, a4, a5, a6);
      if ( v11 )
      {
        v13 = *v8;
        v11[1] = 0;
        *v11 = v13;
        *v8 = (__int64)v11;
        return (__int64)(v11 + 2);
      }
LABEL_8:
      sub_2207530(v10, a2, v12);
    }
    v10 = 4096;
    v15 = (__int64 **)malloc(4096, a2, a3, a4, a5, a6);
    v12 = v15;
    if ( !v15 )
      goto LABEL_8;
    *v15 = v8;
    v8 = (__int64 *)v15;
    v15[1] = 0;
    *(_QWORD *)(a1 + 4096) = v15;
    v9 = ((_DWORD)a2 + 15) & 0xFFFFFFF0;
  }
  v8[1] = v9;
  return *(_QWORD *)(a1 + 4096) - v7 + *(_QWORD *)(*(_QWORD *)(a1 + 4096) + 8LL) + 16;
}

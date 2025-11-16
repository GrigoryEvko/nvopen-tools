// Function: sub_15B89F0
// Address: 0x15b89f0
//
__int64 __fastcall sub_15B89F0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rsi
  int v13; // r8d
  _QWORD *v14; // rdi
  unsigned int v15; // eax
  _QWORD *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h] BYREF
  int v22[12]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = v4 - 1;
    v10 = *(unsigned int *)(*a2 + 8);
    v18 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v19 = *(_QWORD *)(v6 + 8 * (1 - v10));
    v20 = *(_QWORD *)(v6 + 8 * (2 - v10));
    v21 = *(_QWORD *)(v6 + 8 * (3 - v10));
    v22[0] = *(_DWORD *)(v6 + 24);
    v11 = sub_15B3B60(&v18, &v19, &v20, &v21, v22);
    v12 = *a2;
    v13 = 1;
    v14 = 0;
    v15 = v9 & v11;
    v16 = (_QWORD *)(v7 + 8LL * v15);
    v17 = *v16;
    if ( *v16 == *a2 )
    {
LABEL_10:
      *a3 = v16;
      return 1;
    }
    else
    {
      while ( v17 != -8 )
      {
        if ( v17 != -16 || v14 )
          v16 = v14;
        v15 = v9 & (v13 + v15);
        v17 = *(_QWORD *)(v7 + 8LL * v15);
        if ( v17 == v12 )
        {
          v16 = (_QWORD *)(v7 + 8LL * v15);
          goto LABEL_10;
        }
        ++v13;
        v14 = v16;
        v16 = (_QWORD *)(v7 + 8LL * v15);
      }
      if ( !v14 )
        v14 = v16;
      *a3 = v14;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

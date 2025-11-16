// Function: sub_15B77C0
// Address: 0x15b77c0
//
__int64 __fastcall sub_15B77C0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  int v10; // eax
  __int64 v11; // rsi
  int v12; // r8d
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  _QWORD *v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // [rsp+0h] [rbp-40h] BYREF
  __int64 v18; // [rsp+8h] [rbp-38h] BYREF
  bool v19; // [rsp+10h] [rbp-30h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v17 = *(_QWORD *)(*a2 + 24);
    v18 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v19 = *(_DWORD *)(v6 + 4) != 0;
    v9 = v4 - 1;
    v10 = sub_15B62F0(&v17, &v18);
    v11 = *a2;
    v12 = 1;
    v13 = 0;
    v14 = v9 & v10;
    v15 = (_QWORD *)(v7 + 8LL * v14);
    v16 = *v15;
    if ( *v15 == *a2 )
    {
LABEL_10:
      *a3 = v15;
      return 1;
    }
    else
    {
      while ( v16 != -8 )
      {
        if ( v16 != -16 || v13 )
          v15 = v13;
        v14 = v9 & (v12 + v14);
        v16 = *(_QWORD *)(v7 + 8LL * v14);
        if ( v16 == v11 )
        {
          v15 = (_QWORD *)(v7 + 8LL * v14);
          goto LABEL_10;
        }
        ++v12;
        v13 = v15;
        v15 = (_QWORD *)(v7 + 8LL * v14);
      }
      if ( !v13 )
        v13 = v15;
      *a3 = v13;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

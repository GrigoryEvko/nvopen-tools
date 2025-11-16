// Function: sub_15B87F0
// Address: 0x15b87f0
//
__int64 __fastcall sub_15B87F0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // eax
  int v12; // ebx
  int v13; // eax
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // [rsp+0h] [rbp-40h] BYREF
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  int v21[12]; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *(unsigned int *)(*a2 + 8);
    v10 = v6;
    v19 = *(_QWORD *)(v6 + 8 * (1 - v9));
    if ( *(_BYTE *)v6 != 15 )
      v10 = *(_QWORD *)(v6 - 8 * v9);
    v11 = *(_DWORD *)(v6 + 24);
    v20 = v10;
    v12 = v4 - 1;
    v21[0] = v11;
    v13 = sub_15B2A30(&v19, &v20, v21);
    v14 = 1;
    v15 = 0;
    v16 = v12 & v13;
    v17 = (_QWORD *)(v8 + 8LL * v16);
    v18 = *v17;
    if ( *a2 == *v17 )
    {
LABEL_12:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v15 )
          v17 = v15;
        v16 = v12 & (v14 + v16);
        v18 = *(_QWORD *)(v8 + 8LL * v16);
        if ( v18 == *a2 )
        {
          v17 = (_QWORD *)(v8 + 8LL * v16);
          goto LABEL_12;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v8 + 8LL * v16);
      }
      if ( !v15 )
        v15 = v17;
      *a3 = v15;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

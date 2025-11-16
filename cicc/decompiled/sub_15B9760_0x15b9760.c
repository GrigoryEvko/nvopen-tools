// Function: sub_15B9760
// Address: 0x15b9760
//
__int64 __fastcall sub_15B9760(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rcx
  int v11; // eax
  __int64 v12; // rsi
  int v13; // r8d
  _QWORD *v14; // rdi
  unsigned int v15; // eax
  _QWORD *v16; // rcx
  __int64 v17; // rdx
  int v18; // [rsp+0h] [rbp-40h] BYREF
  int v19; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v20; // [rsp+8h] [rbp-38h] BYREF
  __int64 v21[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = v4 - 1;
    v10 = *(unsigned int *)(*a2 + 8);
    v18 = *(unsigned __int16 *)(*a2 + 2);
    v19 = *(_DWORD *)(v6 + 24);
    v20 = *(_QWORD *)(v6 - 8 * v10);
    v21[0] = *(_QWORD *)(v6 + 8 * (1 - v10));
    v11 = sub_15B3100(&v18, &v19, &v20, v21);
    v12 = *a2;
    v13 = 1;
    v14 = 0;
    v15 = v9 & v11;
    v16 = (_QWORD *)(v7 + 8LL * v15);
    v17 = *v16;
    if ( *a2 == *v16 )
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

// Function: sub_15B7230
// Address: 0x15b7230
//
__int64 __fastcall sub_15B7230(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rdx
  __int64 v7; // r14
  int v9; // r13d
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  int v14; // r8d
  _QWORD *v15; // rdi
  unsigned int v16; // eax
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  int v19; // [rsp+Ch] [rbp-54h] BYREF
  __int64 v20; // [rsp+10h] [rbp-50h]
  __int64 v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]
  __int64 v23; // [rsp+28h] [rbp-38h]
  int v24; // [rsp+30h] [rbp-30h]
  int v25; // [rsp+34h] [rbp-2Ch] BYREF
  __int64 v26[5]; // [rsp+38h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v21 = 0;
    v9 = v4 - 1;
    v10 = *(unsigned int *)(v6 + 8);
    v20 = 0;
    v22 = v6 + 8 * (1 - v10);
    v23 = (-8 * (1 - v10)) >> 3;
    v24 = *(_DWORD *)(v6 + 4);
    v25 = *(unsigned __int16 *)(v6 + 2);
    v11 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v19 = v24;
    v26[0] = v11;
    v12 = sub_15B64F0(&v19, &v25, v26);
    v13 = *a2;
    v14 = 1;
    v15 = 0;
    v16 = v9 & v12;
    v17 = (_QWORD *)(v7 + 8LL * v16);
    v18 = *v17;
    if ( *v17 == *a2 )
    {
LABEL_10:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v15 )
          v17 = v15;
        v16 = v9 & (v14 + v16);
        v18 = *(_QWORD *)(v7 + 8LL * v16);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v16);
          goto LABEL_10;
        }
        ++v14;
        v15 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v16);
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

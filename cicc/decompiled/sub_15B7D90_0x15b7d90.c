// Function: sub_15B7D90
// Address: 0x15b7d90
//
__int64 __fastcall sub_15B7D90(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // rdx
  __int64 v10; // rcx
  int v11; // ebx
  int v12; // eax
  __int64 v13; // rsi
  _QWORD *v14; // rdi
  unsigned int v15; // eax
  int v16; // r8d
  _QWORD *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // [rsp+8h] [rbp-98h] BYREF
  __int64 v20; // [rsp+10h] [rbp-90h] BYREF
  int v21; // [rsp+18h] [rbp-88h] BYREF
  __int64 v22; // [rsp+20h] [rbp-80h] BYREF
  __int64 v23[3]; // [rsp+28h] [rbp-78h] BYREF
  int v24; // [rsp+40h] [rbp-60h]
  int v25; // [rsp+44h] [rbp-5Ch]
  __int64 v26; // [rsp+48h] [rbp-58h] BYREF
  int v27; // [rsp+50h] [rbp-50h]
  __int64 v28; // [rsp+58h] [rbp-48h]
  __int64 v29[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(unsigned int *)(*a2 + 8);
    v19 = *(_QWORD *)(*a2 + 8 * (2 - v9));
    v10 = v6;
    if ( *(_BYTE *)v6 != 15 )
      v10 = *(_QWORD *)(v6 - 8LL * *(unsigned int *)(v6 + 8));
    v11 = v4 - 1;
    v20 = v10;
    v21 = *(_DWORD *)(v6 + 24);
    v22 = *(_QWORD *)(v6 + 8 * (1 - v9));
    v23[0] = *(_QWORD *)(v6 + 8 * (3 - v9));
    v23[1] = *(_QWORD *)(v6 + 32);
    v23[2] = *(_QWORD *)(v6 + 40);
    v24 = *(_DWORD *)(v6 + 48);
    v25 = *(_DWORD *)(v6 + 28);
    v26 = *(_QWORD *)(v6 + 8 * (4 - v9));
    v27 = *(_DWORD *)(v6 + 52);
    v28 = *(_QWORD *)(v6 + 8 * (5 - v9));
    v29[0] = *(_QWORD *)(v6 + 8 * (6 - v9));
    v29[1] = *(_QWORD *)(v6 + 8 * (7 - v9));
    v29[2] = *(_QWORD *)(v6 + 8 * (8 - v9));
    v12 = sub_15B5FF0(&v19, &v20, &v21, v23, &v22, &v26, v29);
    v13 = *a2;
    v14 = 0;
    v15 = v11 & v12;
    v16 = 1;
    v17 = (_QWORD *)(v7 + 8LL * v15);
    v18 = *v17;
    if ( *v17 == *a2 )
    {
LABEL_12:
      *a3 = v17;
      return 1;
    }
    else
    {
      while ( v18 != -8 )
      {
        if ( v18 != -16 || v14 )
          v17 = v14;
        v15 = v11 & (v16 + v15);
        v18 = *(_QWORD *)(v7 + 8LL * v15);
        if ( v18 == v13 )
        {
          v17 = (_QWORD *)(v7 + 8LL * v15);
          goto LABEL_12;
        }
        ++v16;
        v14 = v17;
        v17 = (_QWORD *)(v7 + 8LL * v15);
      }
      if ( !v14 )
        v14 = v17;
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

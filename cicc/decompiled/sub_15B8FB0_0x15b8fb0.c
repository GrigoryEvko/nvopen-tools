// Function: sub_15B8FB0
// Address: 0x15b8fb0
//
__int64 __fastcall sub_15B8FB0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // rsi
  _QWORD *v13; // rdi
  unsigned int v14; // eax
  int v15; // r8d
  _QWORD *v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-60h] BYREF
  __int64 v19; // [rsp+8h] [rbp-58h] BYREF
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  int v23; // [rsp+28h] [rbp-38h] BYREF
  int v24[13]; // [rsp+2Ch] [rbp-34h] BYREF

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
    v21 = *(_DWORD *)(v6 + 24);
    v22 = *(_QWORD *)(v6 + 8 * (3 - v10));
    v23 = *(unsigned __int16 *)(v6 + 32);
    v24[0] = *(_DWORD *)(v6 + 36);
    v24[1] = *(_DWORD *)(v6 + 28);
    v11 = sub_15B41C0(&v18, &v19, &v20, &v21, &v22, &v23, v24);
    v12 = *a2;
    v13 = 0;
    v14 = v9 & v11;
    v15 = 1;
    v16 = (_QWORD *)(v7 + 8LL * v14);
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
        if ( v17 != -16 || v13 )
          v16 = v13;
        v14 = v9 & (v15 + v14);
        v17 = *(_QWORD *)(v7 + 8LL * v14);
        if ( v17 == v12 )
        {
          v16 = (_QWORD *)(v7 + 8LL * v14);
          goto LABEL_10;
        }
        ++v15;
        v13 = v16;
        v16 = (_QWORD *)(v7 + 8LL * v14);
      }
      if ( !v13 )
        v13 = v16;
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

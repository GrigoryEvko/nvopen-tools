// Function: sub_15B79C0
// Address: 0x15b79c0
//
__int64 __fastcall sub_15B79C0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  int v9; // ebx
  __int64 v10; // rcx
  __int64 v11; // rdx
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rsi
  int v15; // r8d
  _QWORD *v16; // rdi
  unsigned int v17; // eax
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  int v20; // [rsp+0h] [rbp-50h] BYREF
  __int64 v21[3]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v22; // [rsp+20h] [rbp-30h] BYREF
  int v23[10]; // [rsp+28h] [rbp-28h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = v4 - 1;
    v10 = *(unsigned int *)(*a2 + 8);
    v20 = *(unsigned __int16 *)(*a2 + 2);
    v21[0] = *(_QWORD *)(v6 + 8 * (2 - v10));
    v21[1] = *(_QWORD *)(v6 + 8 * (3 - v10));
    v21[2] = *(_QWORD *)(v6 + 8 * (4 - v10));
    v11 = *(_QWORD *)(v6 + 32);
    v12 = *(_QWORD *)(v6 + 48);
    v22 = v11;
    *(_QWORD *)v23 = v12;
    v13 = sub_15B4F20(&v20, v21, &v22, v23, &v23[1]);
    v14 = *a2;
    v15 = 1;
    v16 = 0;
    v17 = v9 & v13;
    v18 = (_QWORD *)(v7 + 8LL * v17);
    v19 = *v18;
    if ( *v18 == *a2 )
    {
LABEL_10:
      *a3 = v18;
      return 1;
    }
    else
    {
      while ( v19 != -8 )
      {
        if ( v19 != -16 || v16 )
          v18 = v16;
        v17 = v9 & (v15 + v17);
        v19 = *(_QWORD *)(v7 + 8LL * v17);
        if ( v19 == v14 )
        {
          v18 = (_QWORD *)(v7 + 8LL * v17);
          goto LABEL_10;
        }
        ++v15;
        v16 = v18;
        v18 = (_QWORD *)(v7 + 8LL * v17);
      }
      if ( !v16 )
        v16 = v18;
      *a3 = v16;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

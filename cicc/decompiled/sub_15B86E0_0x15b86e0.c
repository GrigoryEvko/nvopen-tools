// Function: sub_15B86E0
// Address: 0x15b86e0
//
__int64 __fastcall sub_15B86E0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v8; // r14
  __int64 v9; // rcx
  __int64 v10; // rsi
  int v11; // edx
  int v12; // eax
  int v13; // ebx
  int v14; // eax
  int v15; // r8d
  _QWORD *v16; // rdi
  unsigned int v17; // eax
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // [rsp+0h] [rbp-40h] BYREF
  __int64 v21; // [rsp+8h] [rbp-38h] BYREF
  int v22; // [rsp+10h] [rbp-30h] BYREF
  int v23[11]; // [rsp+14h] [rbp-2Ch] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *(unsigned int *)(*a2 + 8);
    v10 = v6;
    v20 = *(_QWORD *)(v6 + 8 * (1 - v9));
    if ( *(_BYTE *)v6 != 15 )
      v10 = *(_QWORD *)(v6 - 8 * v9);
    v11 = *(_DWORD *)(v6 + 24);
    v12 = *(unsigned __int16 *)(v6 + 28);
    v21 = v10;
    v13 = v4 - 1;
    v22 = v11;
    v23[0] = v12;
    v14 = sub_15B2700(&v20, &v21, &v22, v23);
    v15 = 1;
    v16 = 0;
    v17 = v13 & v14;
    v18 = (_QWORD *)(v8 + 8LL * v17);
    v19 = *v18;
    if ( *a2 == *v18 )
    {
LABEL_12:
      *a3 = v18;
      return 1;
    }
    else
    {
      while ( v19 != -8 )
      {
        if ( v19 != -16 || v16 )
          v18 = v16;
        v17 = v13 & (v15 + v17);
        v19 = *(_QWORD *)(v8 + 8LL * v17);
        if ( v19 == *a2 )
        {
          v18 = (_QWORD *)(v8 + 8LL * v17);
          goto LABEL_12;
        }
        ++v15;
        v16 = v18;
        v18 = (_QWORD *)(v8 + 8LL * v17);
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

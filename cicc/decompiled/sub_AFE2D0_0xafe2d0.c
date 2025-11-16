// Function: sub_AFE2D0
// Address: 0xafe2d0
//
__int64 __fastcall sub_AFE2D0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // ebx
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  unsigned __int8 v13; // dl
  __int64 *v14; // rcx
  int v15; // ebx
  int v16; // eax
  int v17; // r8d
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  _QWORD *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  int v24[12]; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 )
  {
    v6 = *a2;
    v7 = *(_QWORD *)(a1 + 8);
    v9 = *(_BYTE *)(*a2 - 16);
    v10 = *a2 - 16;
    if ( (v9 & 2) != 0 )
      v11 = *(_QWORD *)(v6 - 32);
    else
      v11 = v10 - 8LL * ((v9 >> 2) & 0xF);
    v22 = *(_QWORD *)(v11 + 8);
    v12 = v6;
    if ( *(_BYTE *)v6 != 16 )
    {
      v13 = *(_BYTE *)(v6 - 16);
      if ( (v13 & 2) != 0 )
        v14 = *(__int64 **)(v6 - 32);
      else
        v14 = (__int64 *)(v10 - 8LL * ((v13 >> 2) & 0xF));
      v12 = *v14;
    }
    v23 = v12;
    v15 = v4 - 1;
    v24[0] = *(_DWORD *)(v6 + 4);
    v16 = sub_AF7750(&v22, &v23, v24);
    v17 = 1;
    v18 = 0;
    v19 = v15 & v16;
    v20 = (_QWORD *)(v7 + 8LL * v19);
    v21 = *v20;
    if ( *a2 == *v20 )
    {
LABEL_18:
      *a3 = v20;
      return 1;
    }
    else
    {
      while ( v21 != -4096 )
      {
        if ( v21 != -8192 || v18 )
          v20 = v18;
        v19 = v15 & (v17 + v19);
        v21 = *(_QWORD *)(v7 + 8LL * v19);
        if ( v21 == *a2 )
        {
          v20 = (_QWORD *)(v7 + 8LL * v19);
          goto LABEL_18;
        }
        ++v17;
        v18 = v20;
        v20 = (_QWORD *)(v7 + 8LL * v19);
      }
      if ( !v18 )
        v18 = v20;
      *a3 = v18;
      return 0;
    }
  }
  else
  {
    *a3 = 0;
    return 0;
  }
}

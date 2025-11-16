// Function: sub_9BA2D0
// Address: 0x9ba2d0
//
__int64 __fastcall sub_9BA2D0(__int64 a1, __int64 a2, int a3, __int64 a4, int a5)
{
  unsigned int v9; // r14d
  __int64 v11; // rax
  int v12; // eax
  __int64 v13; // rsi
  int v14; // ecx
  unsigned int v15; // eax
  __int64 *v16; // rdx
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rcx
  _QWORD *v20; // rax
  _QWORD *v21; // rdx
  __int64 v22; // r8
  unsigned int v23; // edx
  int v24; // r9d
  int v25; // edx
  int v26; // r9d
  _BYTE v27[8]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD *v28; // [rsp+8h] [rbp-58h]
  int v29; // [rsp+14h] [rbp-4Ch]
  unsigned __int8 v30; // [rsp+1Ch] [rbp-44h]
  _BYTE v31[64]; // [rsp+20h] [rbp-40h] BYREF

  if ( !(unsigned __int8)sub_B46490(a2) || !sub_9BA2B0(a3) && !sub_9BA2B0(a5) )
    return 1;
  v11 = *(_QWORD *)(a1 + 32);
  if ( !v11 )
    return 0;
  v9 = *(unsigned __int8 *)(*(_QWORD *)(v11 + 16) + 232LL);
  if ( !(_BYTE)v9 )
    return v9;
  v12 = *(_DWORD *)(a1 + 168);
  v13 = *(_QWORD *)(a1 + 152);
  if ( !v12 )
    return v9;
  v14 = v12 - 1;
  v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (__int64 *)(v13 + 56LL * v15);
  v17 = *v16;
  if ( a2 == *v16 )
  {
LABEL_9:
    sub_C8CD80(v27, v31, v16 + 1);
    v9 = v30;
    if ( !v30 )
    {
      LOBYTE(v9) = sub_C8CA60(v27, a4, v18, v19) == 0;
      if ( !v30 )
        _libc_free(v28, a4);
      return v9;
    }
    v20 = v28;
    v21 = &v28[v29];
    if ( v21 == v28 )
      return v9;
    while ( a4 != *v20 )
    {
      if ( v21 == ++v20 )
        return v9;
    }
    return 0;
  }
  v22 = *v16;
  v23 = v14 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v24 = 1;
  while ( v22 != -4096 )
  {
    v23 = v14 & (v24 + v23);
    v22 = *(_QWORD *)(v13 + 56LL * v23);
    if ( a2 == v22 )
    {
      v25 = 1;
      while ( v17 != -4096 )
      {
        v26 = v25 + 1;
        v15 = v14 & (v25 + v15);
        v16 = (__int64 *)(v13 + 56LL * v15);
        v17 = *v16;
        if ( v22 == *v16 )
          goto LABEL_9;
        v25 = v26;
      }
      return v9;
    }
    ++v24;
  }
  return v9;
}

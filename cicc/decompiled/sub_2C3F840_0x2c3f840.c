// Function: sub_2C3F840
// Address: 0x2c3f840
//
__int64 __fastcall sub_2C3F840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  int v8; // eax
  int v9; // edx
  __int64 v10; // rsi
  int v11; // edi
  _QWORD *v12; // r8
  unsigned int v13; // eax
  _QWORD *v14; // rbx
  __int64 v15; // rcx
  __int64 result; // rax
  int v17; // r12d
  __int64 v18; // rcx
  __int64 v19; // [rsp+50h] [rbp-80h] BYREF
  char *v20; // [rsp+58h] [rbp-78h] BYREF
  __int64 v21; // [rsp+60h] [rbp-70h]
  _BYTE v22[104]; // [rsp+68h] [rbp-68h] BYREF

  v6 = a2 + 96;
  v21 = 0x600000000LL;
  v8 = *(_DWORD *)(a1 + 184);
  v19 = a2 + 96;
  v20 = v22;
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = *(_QWORD *)(a1 + 168);
    v11 = 1;
    v12 = 0;
    v13 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v14 = (_QWORD *)(v10 + 72LL * v13);
    v15 = *v14;
    if ( v6 == *v14 )
      goto LABEL_3;
    while ( v15 != -4096 )
    {
      if ( !v12 && v15 == -8192 )
        v12 = v14;
      a6 = (unsigned int)(v11 + 1);
      v13 = v9 & (v11 + v13);
      v14 = (_QWORD *)(v10 + 72LL * v13);
      v15 = *v14;
      if ( v6 == *v14 )
        goto LABEL_3;
      ++v11;
    }
    if ( !v12 )
      v12 = v14;
  }
  else
  {
    v12 = 0;
  }
  v14 = sub_2C3F6F0(a1 + 160, &v19, v12);
  *v14 = v19;
  v14[1] = v14 + 3;
  v14[2] = 0x600000000LL;
  if ( (_DWORD)v21 )
    sub_2C3D860((__int64)(v14 + 1), &v20, (unsigned int)v21, v18, (__int64)v12, a6);
  if ( v20 != v22 )
    _libc_free((unsigned __int64)v20);
LABEL_3:
  result = *(unsigned int *)(a1 + 8);
  if ( (_DWORD)result )
  {
    result = *((unsigned int *)v14 + 4);
    v17 = 0;
    do
    {
      if ( result + 1 > (unsigned __int64)*((unsigned int *)v14 + 5) )
      {
        sub_C8D5F0((__int64)(v14 + 1), v14 + 3, result + 1, 8u, (__int64)v12, a6);
        result = *((unsigned int *)v14 + 4);
      }
      ++v17;
      *(_QWORD *)(v14[1] + 8 * result) = v6;
      result = (unsigned int)(*((_DWORD *)v14 + 4) + 1);
      *((_DWORD *)v14 + 4) = result;
    }
    while ( *(_DWORD *)(a1 + 8) != v17 );
  }
  return result;
}

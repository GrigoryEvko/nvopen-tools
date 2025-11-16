// Function: sub_E11E80
// Address: 0xe11e80
//
void *__fastcall sub_E11E80(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  _BYTE *v8; // rdx
  __int64 **v9; // rcx
  _BYTE *v10; // rsi
  size_t v11; // r15
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rax
  __int64 **v14; // rdi
  _QWORD *v15; // rax
  __int64 v16; // rdi
  void *v17; // r8
  __int64 **v18; // rax
  __int64 *v20; // [rsp+8h] [rbp-48h]
  _BYTE *v21; // [rsp+10h] [rbp-40h]

  v6 = 8 * a2;
  v8 = (_BYTE *)a1[3];
  v9 = (__int64 **)a1[614];
  v10 = (_BYTE *)(8 * a2 + a1[2]);
  v11 = v8 - v10;
  v12 = ((_DWORD)v8 - (_DWORD)v10 + 15) & 0xFFFFFFF0;
  v13 = (unsigned __int64)v9[1] + v12;
  if ( v13 > 0xFEF )
  {
    v20 = (__int64 *)a1[614];
    v21 = (_BYTE *)a1[3];
    if ( v12 > 0xFF0 )
    {
      v14 = (__int64 **)(v12 + 16);
      v15 = (_QWORD *)malloc(v12 + 16, v10, v8, v9, a5, a6);
      v8 = v21;
      if ( v15 )
      {
        v16 = *v20;
        v17 = v15 + 2;
        v15[1] = 0;
        *v15 = v16;
        *v20 = (__int64)v15;
        goto LABEL_8;
      }
LABEL_11:
      sub_2207530(v14, v10, v8);
    }
    v18 = (__int64 **)malloc(4096, v10, v8, v9, a5, a6);
    v14 = v18;
    if ( !v18 )
      goto LABEL_11;
    v8 = v21;
    v18[1] = 0;
    a1[614] = v18;
    *v18 = v20;
    v13 = ((_DWORD)v11 + 15) & 0xFFFFFFF0;
    v9 = v14;
  }
  v9[1] = (__int64 *)v13;
  v17 = (void *)(a1[614] - v12 + *(_QWORD *)(a1[614] + 8LL) + 16);
LABEL_8:
  if ( v10 != v8 )
    v17 = memmove(v17, v10, v11);
  a1[3] = a1[2] + v6;
  return v17;
}

// Function: sub_2DDB890
// Address: 0x2ddb890
//
void __fastcall sub_2DDB890(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r14
  _DWORD *v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rax
  _DWORD **v17; // r15
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v23; // [rsp+10h] [rbp-70h]
  __int64 v24; // [rsp+10h] [rbp-70h]
  __int64 v25[2]; // [rsp+20h] [rbp-60h] BYREF
  _BYTE v26[80]; // [rsp+30h] [rbp-50h] BYREF

  v7 = a1 + 48 * a2;
  v8 = (a3 - 1) / 2;
  v23 = (__int64)a4;
  if ( a2 >= v8 )
  {
    v9 = a2;
    v13 = v7;
  }
  else
  {
    v9 = a2;
    while ( 1 )
    {
      v10 = v9 + 1;
      v9 = 2 * (v9 + 1);
      v11 = a1 + 16 * (v9 + 4 * v10);
      v12 = *(_DWORD **)(v11 - 48);
      if ( **(_DWORD **)v11 < *v12 || **(_DWORD **)v11 == *v12 && *(_DWORD *)(*(_QWORD *)v11 + 4LL) < v12[1] )
      {
        --v9;
        v11 = a1 + 48 * v9;
      }
      v13 = v11;
      sub_2DDB710(v7, v11, (__int64)v12, (__int64)a4, a5, a6);
      if ( v9 >= v8 )
        break;
      v7 = v11;
    }
  }
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v9 )
  {
    v19 = v9 + 1;
    v20 = 2 * (v9 + 1);
    v21 = v20 + 4 * v19;
    v9 = v20 - 1;
    sub_2DDB710(v13, a1 + 16 * v21 - 48, a3, v19, a5, a6);
    v13 = a1 + 48 * v9;
  }
  v25[1] = 0x400000000LL;
  v25[0] = (__int64)v26;
  v14 = *(unsigned int *)(v23 + 8);
  if ( (_DWORD)v14 )
    sub_2DDB710((__int64)v25, v23, v14, (__int64)a4, a5, a6);
  v15 = v9 - 1;
  v16 = (v9 - 1) / 2;
  if ( v9 > a2 )
  {
    while ( 1 )
    {
      v15 = v25[0];
      v17 = (_DWORD **)(a1 + 48 * v16);
      a4 = *v17;
      if ( **v17 >= *(_DWORD *)v25[0] && (**v17 != *(_DWORD *)v25[0] || a4[1] >= *(_DWORD *)(v25[0] + 4)) )
        break;
      v18 = v13;
      v24 = v16;
      v13 = a1 + 48 * v16;
      sub_2DDB710(v18, v13, v25[0], (__int64)a4, a5, a6);
      a4 = (_DWORD *)(v24 - 1);
      v15 = (v24 - 1) / 2;
      if ( a2 >= v24 )
        break;
      v16 = (v24 - 1) / 2;
    }
  }
  sub_2DDB710(v13, (__int64)v25, v15, (__int64)a4, a5, a6);
  if ( (_BYTE *)v25[0] != v26 )
    _libc_free(v25[0]);
}

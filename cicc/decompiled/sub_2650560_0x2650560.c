// Function: sub_2650560
// Address: 0x2650560
//
__int64 __fastcall sub_2650560(_QWORD *a1, __int64 *a2, __int64 a3)
{
  unsigned __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rdi
  _BYTE *v11; // rsi
  __int64 v12; // r9
  __int64 v13; // rdx
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rcx
  int v16; // r8d
  __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v20; // rsi
  __int64 v21; // rdx
  _QWORD *v22; // rax
  __int64 v23; // rcx
  _QWORD *v24; // rdx
  __int64 v25; // rsi
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // [rsp+0h] [rbp-70h]
  int v31; // [rsp+8h] [rbp-68h]
  int v32; // [rsp+8h] [rbp-68h]
  unsigned __int64 v33; // [rsp+8h] [rbp-68h]
  unsigned __int64 v34; // [rsp+8h] [rbp-68h]
  unsigned __int64 v35; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v36[10]; // [rsp+20h] [rbp-50h] BYREF

  v35 = *(_QWORD *)*a2;
  v6 = v35;
  v7 = (__int64 *)sub_263EE40(a1 + 6, &v35);
  v8 = sub_2648220(a1, *(_BYTE *)v35, *v7, *(_QWORD *)(v6 + 8), *(_QWORD *)(v6 + 16));
  v9 = v35;
  v36[0] = v8;
  v10 = *(_QWORD *)(v35 + 120);
  if ( v10 )
  {
    v11 = *(_BYTE **)(v10 + 104);
    if ( v11 == *(_BYTE **)(v10 + 112) )
    {
      v33 = v35;
      sub_263EEC0(v10 + 96, v11, v36);
      v13 = v36[0];
      v12 = v35;
      v9 = v33;
    }
    else
    {
      v12 = v35;
      if ( v11 )
      {
        *(_QWORD *)v11 = v8;
        v11 = *(_BYTE **)(v10 + 104);
        v12 = v35;
      }
      v13 = v8;
      *(_QWORD *)(v10 + 104) = v11 + 8;
    }
    *(_QWORD *)(v13 + 120) = *(_QWORD *)(v9 + 120);
  }
  else
  {
    v20 = *(_BYTE **)(v35 + 104);
    if ( v20 == *(_BYTE **)(v35 + 112) )
    {
      v34 = v35;
      sub_263EEC0(v35 + 96, v20, v36);
      v21 = v36[0];
      v12 = v35;
      v9 = v34;
    }
    else
    {
      v12 = v35;
      if ( v20 )
      {
        *(_QWORD *)v20 = v8;
        v20 = *(_BYTE **)(v9 + 104);
        v12 = v35;
      }
      v21 = v8;
      *(_QWORD *)(v9 + 104) = v20 + 8;
    }
    *(_QWORD *)(v21 + 120) = v9;
  }
  if ( v8 + 24 != v12 + 24 )
  {
    v14 = *(unsigned int *)(v12 + 32);
    v15 = *(unsigned int *)(v8 + 32);
    v16 = *(_DWORD *)(v12 + 32);
    if ( v14 <= v15 )
    {
      if ( *(_DWORD *)(v12 + 32) )
      {
        v22 = *(_QWORD **)(v12 + 24);
        v23 = *(_QWORD *)(v8 + 24);
        v24 = &v22[2 * v14];
        do
        {
          v25 = *v22;
          v22 += 2;
          v23 += 16;
          *(_QWORD *)(v23 - 16) = v25;
          *(_DWORD *)(v23 - 8) = *((_DWORD *)v22 - 2);
        }
        while ( v22 != v24 );
      }
    }
    else
    {
      if ( v14 > *(unsigned int *)(v8 + 36) )
      {
        v30 = v12;
        *(_DWORD *)(v8 + 32) = 0;
        v32 = v14;
        sub_C8D5F0(v8 + 24, (const void *)(v8 + 40), v14, 0x10u, v14, v12);
        v12 = v30;
        v16 = v32;
        v15 = 0;
        v14 = *(unsigned int *)(v30 + 32);
      }
      else if ( *(_DWORD *)(v8 + 32) )
      {
        v26 = *(_QWORD **)(v12 + 24);
        v15 *= 16LL;
        v27 = *(_QWORD *)(v8 + 24);
        v28 = (_QWORD *)((char *)v26 + v15);
        do
        {
          v29 = *v26;
          v26 += 2;
          v27 += 16;
          *(_QWORD *)(v27 - 16) = v29;
          *(_DWORD *)(v27 - 8) = *((_DWORD *)v26 - 2);
        }
        while ( v26 != v28 );
        v14 = *(unsigned int *)(v12 + 32);
      }
      v17 = *(_QWORD *)(v12 + 24);
      v18 = 16 * v14;
      if ( v17 + v15 != v18 + v17 )
      {
        v31 = v16;
        memcpy((void *)(v15 + *(_QWORD *)(v8 + 24)), (const void *)(v17 + v15), v18 - v15);
        v16 = v31;
      }
    }
    *(_DWORD *)(v8 + 32) = v16;
  }
  memset(v36, 0, 32);
  sub_264A680((__int64)v36, a3);
  sub_264FE30((__int64)a1, a2, v8, 1, (__int64)v36);
  sub_2342640((__int64)v36);
  return v8;
}

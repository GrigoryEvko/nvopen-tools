// Function: sub_256B2D0
// Address: 0x256b2d0
//
bool __fastcall sub_256B2D0(__int64 *a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rdi
  __int64 v5; // r8
  __int64 v6; // rsi
  int v7; // eax
  __int64 v8; // r9
  int v9; // ecx
  unsigned int v11; // eax
  __int64 v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r12
  __int64 *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdi
  __int64 v19; // rcx
  bool result; // al
  int v21; // r10d
  __int64 *v22; // rbx
  __int64 *v23; // r12
  __int64 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // [rsp+0h] [rbp-30h] BYREF
  __int64 v29[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = *a1;
  v5 = *a2;
  v6 = *a3;
  v7 = *(_DWORD *)(v4 + 24);
  v8 = *(_QWORD *)(v4 + 8);
  if ( !v7 )
  {
LABEL_11:
    v22 = (__int64 *)a1[1];
    v29[0] = v5;
    v28 = v6;
    v23 = sub_256A2C0(*v22, &v28);
    v24 = sub_256A2C0(*v22, v29);
    sub_256B200((__int64)v23, (__int64)v24, v25, v26, v27);
    return 1;
  }
  v9 = v7 - 1;
  v11 = (v7 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = *(_QWORD *)(v8 + 104LL * v11);
  if ( v12 != v6 )
  {
    v21 = 1;
    while ( v12 != -4096 )
    {
      v11 = v9 & (v21 + v11);
      v12 = *(_QWORD *)(v8 + 104LL * v11);
      if ( v6 == v12 )
        goto LABEL_3;
      ++v21;
    }
    goto LABEL_11;
  }
LABEL_3:
  v28 = v5;
  v13 = sub_256A430(v4, &v28);
  v14 = *a1;
  v15 = (__int64)v13;
  v29[0] = *a3;
  v16 = sub_256A430(v14, v29);
  v17 = v16[11];
  v18 = (__int64)v16;
  if ( !v17 )
    v17 = *((unsigned int *)v16 + 2);
  v19 = *(_QWORD *)(v15 + 88);
  if ( !v19 )
    v19 = *(unsigned int *)(v15 + 8);
  result = 0;
  if ( v19 == v17 )
    return sub_255D700(v18, v15);
  return result;
}

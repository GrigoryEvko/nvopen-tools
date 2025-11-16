// Function: sub_2282310
// Address: 0x2282310
//
__int64 __fastcall sub_2282310(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v6; // rax
  __int64 v7; // r9
  unsigned int v9; // eax
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // esi
  unsigned int v13; // edi
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // r14
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  _QWORD *v25; // rax
  unsigned __int64 v26; // r12
  _QWORD *v27; // r13
  __int64 v28; // rdx
  __int64 v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  int v32; // r13d
  int v33; // eax
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h] BYREF
  __int64 v36; // [rsp+18h] [rbp-48h] BYREF
  __int64 v37; // [rsp+20h] [rbp-40h] BYREF
  int v38; // [rsp+28h] [rbp-38h]

  v6 = *a2;
  v38 = 0;
  v37 = v6;
  if ( (unsigned __int8)sub_227C5D0(a1, &v37, &v35) )
    return *(_QWORD *)(a1 + 272) + 32LL * *(unsigned int *)(v35 + 8);
  v9 = *(_DWORD *)(a1 + 8);
  v10 = v35;
  ++*(_QWORD *)a1;
  v36 = v10;
  v11 = (v9 >> 1) + 1;
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v13 = 48;
    v12 = 16;
  }
  else
  {
    v12 = *(_DWORD *)(a1 + 24);
    v13 = 3 * v12;
  }
  if ( 4 * v11 >= v13 )
  {
    v12 *= 2;
    goto LABEL_19;
  }
  if ( v12 - (v11 + *(_DWORD *)(a1 + 12)) <= v12 >> 3 )
  {
LABEL_19:
    sub_2281F90(a1, v12);
    sub_227C5D0(a1, &v37, &v36);
    v10 = v36;
    v11 = (*(_DWORD *)(a1 + 8) >> 1) + 1;
  }
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a1 + 8) & 1 | (2 * v11);
  if ( *(_QWORD *)v10 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v10 = v37;
  *(_DWORD *)(v10 + 8) = v38;
  *(_DWORD *)(v10 + 8) = *(_DWORD *)(a1 + 280);
  v14 = *(unsigned int *)(a1 + 280);
  v15 = v14;
  if ( *(_DWORD *)(a1 + 284) <= (unsigned int)v14 )
  {
    v16 = sub_C8D7D0(a1 + 272, a1 + 288, 0, 0x20u, (unsigned __int64 *)&v37, v7);
    v21 = 4LL * *(unsigned int *)(a1 + 280);
    v22 = (__int64 *)(v21 * 8 + v16);
    if ( v21 * 8 + v16 )
    {
      v23 = *a2;
      v22[1] = 6;
      v22[2] = 0;
      *v22 = v23;
      v24 = a3[2];
      v22[3] = v24;
      if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
        sub_BD6050((unsigned __int64 *)v22 + 1, *a3 & 0xFFFFFFFFFFFFFFF8LL);
      v21 = 4LL * *(unsigned int *)(a1 + 280);
    }
    v25 = *(_QWORD **)(a1 + 272);
    v26 = (unsigned __int64)&v25[v21];
    if ( v25 != &v25[v21] )
    {
      v27 = (_QWORD *)v16;
      do
      {
        if ( v27 )
        {
          v28 = *v25;
          v27[1] = 6;
          v27[2] = 0;
          *v27 = v28;
          v29 = v25[3];
          v27[3] = v29;
          if ( v29 != 0 && v29 != -4096 && v29 != -8192 )
          {
            v34 = v25;
            sub_BD6050(v27 + 1, v25[1] & 0xFFFFFFFFFFFFFFF8LL);
            v25 = v34;
          }
        }
        v25 += 4;
        v27 += 4;
      }
      while ( (_QWORD *)v26 != v25 );
      v30 = *(_QWORD *)(a1 + 272);
      v26 = v30 + 32LL * *(unsigned int *)(a1 + 280);
      if ( v30 != v26 )
      {
        do
        {
          v31 = *(_QWORD *)(v26 - 8);
          v26 -= 32LL;
          if ( v31 != 0 && v31 != -4096 && v31 != -8192 )
            sub_BD60C0((_QWORD *)(v26 + 8));
        }
        while ( v26 != v30 );
        v26 = *(_QWORD *)(a1 + 272);
      }
    }
    v32 = v37;
    if ( a1 + 288 != v26 )
      _libc_free(v26);
    v33 = *(_DWORD *)(a1 + 280);
    *(_QWORD *)(a1 + 272) = v16;
    *(_DWORD *)(a1 + 284) = v32;
    v20 = (unsigned int)(v33 + 1);
    *(_DWORD *)(a1 + 280) = v20;
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 272);
    v17 = (__int64 *)(v16 + 32 * v14);
    if ( v17 )
    {
      v18 = *a2;
      v17[1] = 6;
      v17[2] = 0;
      *v17 = v18;
      v19 = a3[2];
      v17[3] = v19;
      if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
        sub_BD6050((unsigned __int64 *)v17 + 1, *a3 & 0xFFFFFFFFFFFFFFF8LL);
      v15 = *(_DWORD *)(a1 + 280);
      v16 = *(_QWORD *)(a1 + 272);
    }
    v20 = (unsigned int)(v15 + 1);
    *(_DWORD *)(a1 + 280) = v20;
  }
  return v16 + 32 * v20 - 32;
}

// Function: sub_DB30A0
// Address: 0xdb30a0
//
__int64 __fastcall sub_DB30A0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *v7; // r15
  _QWORD *v8; // rbx
  _QWORD *i; // r13
  char v10; // al
  __int64 v11; // rax
  bool v12; // zf
  _QWORD *j; // rbx
  __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // r13d
  __int64 v17; // rdi
  int v18; // ecx
  unsigned int v19; // esi
  __int64 v20; // rdx
  _QWORD *v21; // r13
  __int64 v22; // r10
  __int64 v23; // r9
  __int64 v24; // rsi
  __int64 v25; // rcx
  unsigned __int64 *v26; // r8
  __int64 result; // rax
  _QWORD *v28; // rbx
  _QWORD *k; // r12
  char v30; // al
  __int64 v31; // rax
  _QWORD *v32; // rax
  int v33; // edx
  _QWORD *v34; // r11
  __int64 v35; // [rsp+8h] [rbp-118h]
  __int64 v36; // [rsp+20h] [rbp-100h]
  void *v37; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-E8h] BYREF
  __int64 v39; // [rsp+48h] [rbp-D8h]
  void *v40; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v41; // [rsp+68h] [rbp-B8h] BYREF
  __int64 v42; // [rsp+78h] [rbp-A8h]
  void *v43; // [rsp+90h] [rbp-90h] BYREF
  __int64 v44; // [rsp+98h] [rbp-88h] BYREF
  __int64 v45; // [rsp+A8h] [rbp-78h]
  void *v46; // [rsp+C0h] [rbp-60h] BYREF
  _QWORD v47[2]; // [rsp+C8h] [rbp-58h] BYREF
  __int64 v48; // [rsp+D8h] [rbp-48h]
  __int64 v49; // [rsp+E0h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 8) = sub_C7D670(48LL * v6, 8);
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v35 = 48 * v4;
    v7 = (_QWORD *)(48 * v4 + v5);
    sub_D982A0(&v46, -4096, 0);
    v8 = *(_QWORD **)(a1 + 8);
    for ( i = &v8[6 * *(unsigned int *)(a1 + 24)]; i != v8; v8 += 6 )
    {
      if ( v8 )
      {
        v10 = v47[0];
        v8[2] = 0;
        v8[1] = v10 & 6;
        v11 = v48;
        v12 = v48 == -4096;
        v8[3] = v48;
        if ( v11 != 0 && !v12 && v11 != -8192 )
          sub_BD6050(v8 + 1, v47[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v8 = &unk_49DE910;
        v8[4] = v49;
      }
    }
    v46 = &unk_49DB368;
    if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
      sub_BD60C0(v47);
    sub_D982A0(&v37, -4096, 0);
    sub_D982A0(&v40, -8192, 0);
    for ( j = (_QWORD *)v5; v7 != j; j += 6 )
    {
      v14 = j[3];
      if ( v39 == v14 )
      {
        v15 = j[3];
      }
      else
      {
        v15 = v42;
        if ( v14 != v42 )
        {
          v16 = *(_DWORD *)(a1 + 24);
          if ( !v16 )
            BUG();
          v36 = *(_QWORD *)(a1 + 8);
          sub_D982A0(&v43, -4096, 0);
          sub_D982A0(&v46, -8192, 0);
          v17 = j[3];
          v18 = v16 - 1;
          v19 = (v16 - 1) & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v20 = 48LL * v19;
          v21 = (_QWORD *)(v36 + v20);
          v22 = *(_QWORD *)(v36 + v20 + 24);
          if ( v22 == v17 )
          {
            v23 = v48;
          }
          else
          {
            v23 = v48;
            v32 = (_QWORD *)(v36 + v20);
            v33 = 1;
            v21 = 0;
            while ( v45 != v22 )
            {
              if ( v48 != v22 || v21 )
                v32 = v21;
              v19 = v18 & (v33 + v19);
              v21 = (_QWORD *)(v36 + 48LL * v19);
              v22 = v21[3];
              if ( v17 == v22 )
                goto LABEL_20;
              ++v33;
              v34 = v32;
              v32 = (_QWORD *)(v36 + 48LL * v19);
              v21 = v34;
            }
            if ( !v21 )
              v21 = v32;
          }
LABEL_20:
          v46 = &unk_49DB368;
          if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
            sub_BD60C0(v47);
          v43 = &unk_49DB368;
          if ( v45 != -4096 && v45 != 0 && v45 != -8192 )
            sub_BD60C0(&v44);
          v24 = v21[3];
          v25 = j[3];
          v26 = v21 + 1;
          if ( v24 != v25 )
          {
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
            {
              sub_BD60C0(v21 + 1);
              v25 = j[3];
              v26 = v21 + 1;
            }
            v21[3] = v25;
            if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
              sub_BD6050(v26, j[1] & 0xFFFFFFFFFFFFFFF8LL);
          }
          v21[4] = j[4];
          v21[5] = j[5];
          ++*(_DWORD *)(a1 + 16);
          v15 = j[3];
        }
      }
      *j = &unk_49DB368;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD60C0(j + 1);
    }
    v40 = &unk_49DB368;
    if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
      sub_BD60C0(&v41);
    v37 = &unk_49DB368;
    if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
      sub_BD60C0(&v38);
    return sub_C7D6A0(v5, v35, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    sub_D982A0(&v46, -4096, 0);
    v28 = *(_QWORD **)(a1 + 8);
    for ( k = &v28[6 * *(unsigned int *)(a1 + 24)]; k != v28; v28 += 6 )
    {
      if ( v28 )
      {
        v30 = v47[0];
        v28[2] = 0;
        v28[1] = v30 & 6;
        v31 = v48;
        v12 = v48 == 0;
        v28[3] = v48;
        if ( v31 != -4096 && !v12 && v31 != -8192 )
          sub_BD6050(v28 + 1, v47[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v28 = &unk_49DE910;
        v28[4] = v49;
      }
    }
    v46 = &unk_49DB368;
    result = v48;
    if ( v48 != 0 && v48 != -4096 && v48 != -8192 )
      return sub_BD60C0(v47);
  }
  return result;
}

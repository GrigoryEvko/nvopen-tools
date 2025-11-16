// Function: sub_D98440
// Address: 0xd98440
//
void __fastcall sub_D98440(__int64 a1, __int64 a2)
{
  int v2; // r12d
  __int64 v3; // r13
  __int64 v5; // rdx
  int v6; // eax
  unsigned int v7; // edi
  _QWORD *v8; // r12
  __int64 v9; // rsi
  unsigned int v10; // esi
  __int64 v11; // rdi
  __int64 v12; // r8
  unsigned int v13; // eax
  __int64 *v14; // r13
  __int64 v15; // r10
  _QWORD *v16; // rdi
  __int64 v17; // rsi
  _QWORD *v18; // rax
  int v19; // r8d
  const void *v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rdi
  int v25; // eax
  unsigned int v26; // r15d
  __int64 *v27; // rsi
  __int64 v28; // r8
  __int64 v29; // rax
  _QWORD *v30; // rdi
  int v31; // r8d
  int v32; // ecx
  int v33; // esi
  int v34; // r9d
  __int64 v35; // [rsp+8h] [rbp-68h] BYREF
  void *v36; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v37[2]; // [rsp+18h] [rbp-58h] BYREF
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39; // [rsp+30h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 152);
  v35 = a2;
  if ( !v2 )
    return;
  v3 = *(_QWORD *)(a1 + 136);
  sub_D982A0(&v36, -4096, 0);
  v5 = v35;
  v6 = v2 - 1;
  v7 = (v2 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
  v8 = (_QWORD *)(v3 + 48LL * v7);
  v9 = v8[3];
  if ( v35 == v9 )
  {
LABEL_3:
    v36 = &unk_49DB368;
    if ( v38 && v38 != -8192 && v38 != -4096 )
    {
      sub_BD60C0(v37);
      v5 = v35;
    }
    if ( v8 != (_QWORD *)(*(_QWORD *)(a1 + 136) + 48LL * *(unsigned int *)(a1 + 152)) )
    {
      v10 = *(_DWORD *)(a1 + 120);
      v11 = v8[5];
      v12 = *(_QWORD *)(a1 + 104);
      if ( v10 )
      {
        v13 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v14 = (__int64 *)(v12 + 88LL * v13);
        v15 = *v14;
        if ( v11 == *v14 )
        {
LABEL_10:
          if ( !*((_DWORD *)v14 + 6) )
            goto LABEL_11;
          goto LABEL_29;
        }
        v32 = 1;
        while ( v15 != -4096 )
        {
          v13 = (v10 - 1) & (v32 + v13);
          v14 = (__int64 *)(v12 + 88LL * v13);
          v15 = *v14;
          if ( v11 == *v14 )
            goto LABEL_10;
          ++v32;
        }
      }
      v14 = (__int64 *)(v12 + 88LL * v10);
      if ( !*((_DWORD *)v14 + 6) )
      {
LABEL_11:
        v16 = (_QWORD *)v14[5];
        v17 = (__int64)&v16[*((unsigned int *)v14 + 12)];
        v18 = sub_D91230(v16, v17, &v35);
        if ( (_QWORD *)v17 == v18 )
        {
LABEL_15:
          sub_D982A0(&v36, -8192, 0);
          v21 = v8[3];
          v22 = v38;
          if ( v21 != v38 )
          {
            if ( v21 != -4096 && v21 != 0 && v21 != -8192 )
            {
              sub_BD60C0(v8 + 1);
              v22 = v38;
            }
            v8[3] = v22;
            if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
              sub_BD6050(v8 + 1, v37[0] & 0xFFFFFFFFFFFFFFF8LL);
            v22 = v38;
          }
          v8[4] = v39;
          v36 = &unk_49DB368;
          if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
            sub_BD60C0(v37);
          --*(_DWORD *)(a1 + 144);
          ++*(_DWORD *)(a1 + 148);
          return;
        }
        v20 = v18 + 1;
        if ( (_QWORD *)v17 == v18 + 1 )
        {
LABEL_14:
          *((_DWORD *)v14 + 12) = v19 - 1;
          goto LABEL_15;
        }
LABEL_13:
        memmove(v18, v20, v17 - (_QWORD)v20);
        v19 = *((_DWORD *)v14 + 12);
        goto LABEL_14;
      }
LABEL_29:
      v23 = *((_DWORD *)v14 + 8);
      v24 = v14[2];
      if ( !v23 )
        goto LABEL_15;
      v25 = v23 - 1;
      v26 = v25 & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
      v27 = (__int64 *)(v24 + 8LL * v26);
      v28 = *v27;
      if ( v5 != *v27 )
      {
        v33 = 1;
        while ( v28 != -4096 )
        {
          v34 = v33 + 1;
          v26 = v25 & (v33 + v26);
          v27 = (__int64 *)(v24 + 8LL * v26);
          v28 = *v27;
          if ( v5 == *v27 )
            goto LABEL_31;
          v33 = v34;
        }
        goto LABEL_15;
      }
LABEL_31:
      *v27 = -8192;
      v29 = *((unsigned int *)v14 + 12);
      --*((_DWORD *)v14 + 6);
      v30 = (_QWORD *)v14[5];
      ++*((_DWORD *)v14 + 7);
      v17 = (__int64)&v30[v29];
      v18 = sub_D91230(v30, v17, &v35);
      v20 = v18 + 1;
      if ( v18 + 1 == (_QWORD *)v17 )
        goto LABEL_14;
      goto LABEL_13;
    }
  }
  else
  {
    v31 = 1;
    while ( v38 != v9 )
    {
      v7 = v6 & (v31 + v7);
      v8 = (_QWORD *)(v3 + 48LL * v7);
      v9 = v8[3];
      if ( v35 == v9 )
        goto LABEL_3;
      ++v31;
    }
    if ( v38 && v38 != -4096 && v38 != -8192 )
    {
      v36 = &unk_49DB368;
      sub_BD60C0(v37);
    }
  }
}

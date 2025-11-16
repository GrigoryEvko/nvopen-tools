// Function: sub_CFF330
// Address: 0xcff330
//
_QWORD *__fastcall sub_CFF330(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r12
  _QWORD *v4; // r15
  unsigned int v5; // eax
  _QWORD *result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // rcx
  char v9; // dl
  __int64 v10; // rcx
  _QWORD *v11; // r14
  __int64 v12; // rax
  int v13; // edx
  __int64 v14; // rdi
  int v15; // ecx
  unsigned int v16; // edx
  _QWORD *v17; // rbx
  unsigned __int64 v18; // rsi
  __int64 v19; // r13
  __int64 v20; // rax
  __int64 v21; // rsi
  _QWORD *v22; // rbx
  _QWORD *v23; // r15
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // r15
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v29; // rcx
  _QWORD *v30; // r14
  __int64 v31; // rdi
  _QWORD *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rdx
  _QWORD *j; // rcx
  char v36; // dl
  int v37; // r10d
  _QWORD *v38; // r9
  __int64 v39; // rdx
  _QWORD *v40; // [rsp+10h] [rbp-120h]
  __int64 v41; // [rsp+18h] [rbp-118h]
  __int64 v42; // [rsp+20h] [rbp-110h]
  __int64 v43; // [rsp+28h] [rbp-108h]
  _QWORD *v45; // [rsp+38h] [rbp-F8h]
  _QWORD v46[2]; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v47; // [rsp+58h] [rbp-D8h]
  __int64 v48; // [rsp+60h] [rbp-D0h]
  void *v49; // [rsp+70h] [rbp-C0h]
  _QWORD v50[2]; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v51; // [rsp+88h] [rbp-A8h]
  __int64 v52; // [rsp+90h] [rbp-A0h]
  void *v53; // [rsp+A0h] [rbp-90h]
  _QWORD v54[2]; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-78h]
  __int64 v56; // [rsp+C0h] [rbp-70h]
  void *v57; // [rsp+D0h] [rbp-60h]
  __int64 v58; // [rsp+D8h] [rbp-58h] BYREF
  __int64 v59; // [rsp+E0h] [rbp-50h]
  __int64 v60; // [rsp+E8h] [rbp-48h]
  __int64 i; // [rsp+F0h] [rbp-40h]

  v2 = (unsigned int)(a2 - 1);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
  v43 = (__int64)v4;
  v5 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
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
  if ( v5 < 0x40 )
    v5 = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_C7D670(48LL * v5, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v58 = 2;
    v59 = 0;
    v42 = 48 * v3;
    v7 = *(unsigned int *)(a1 + 24);
    v45 = &v4[6 * v3];
    *(_QWORD *)(a1 + 16) = 0;
    v60 = -4096;
    v8 = &result[6 * v7];
    v57 = &unk_49DDB10;
    for ( i = 0; v8 != result; result += 6 )
    {
      if ( result )
      {
        v9 = v58;
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DDB10;
        result[1] = v9 & 6;
        result[4] = i;
      }
    }
    v46[0] = 2;
    v46[1] = 0;
    v47 = -4096;
    v48 = 0;
    v50[0] = 2;
    v50[1] = 0;
    v51 = -8192;
    v49 = &unk_49DDB10;
    v52 = 0;
    if ( v45 != v4 )
    {
      v10 = -4096;
      v11 = v4;
      v12 = v4[3];
      if ( v12 == -4096 )
        goto LABEL_25;
LABEL_10:
      v10 = v51;
      if ( v12 == v51 )
        goto LABEL_25;
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
        BUG();
      v14 = *(_QWORD *)(a1 + 8);
      v15 = v13 - 1;
      v16 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v17 = (_QWORD *)(v14 + 48LL * v16);
      v18 = v17[3];
      if ( v12 != v18 )
      {
        v37 = 1;
        v38 = 0;
        while ( v18 != -4096 )
        {
          if ( !v38 && v18 == -8192 )
            v38 = v17;
          v16 = v15 & (v37 + v16);
          v17 = (_QWORD *)(v14 + 48LL * v16);
          v18 = v17[3];
          if ( v12 == v18 )
            goto LABEL_13;
          ++v37;
        }
        if ( v38 )
        {
          v39 = v38[3];
          v17 = v38;
        }
        else
        {
          v39 = v17[3];
        }
        if ( v12 != v39 )
        {
          LOBYTE(v18) = v39 != 0;
          if ( v39 != -4096 && v39 != 0 && v39 != -8192 )
          {
            sub_BD60C0(v17 + 1);
            v12 = v11[3];
          }
          v17[3] = v12;
          if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
          {
            v18 = v11[1] & 0xFFFFFFFFFFFFFFF8LL;
            sub_BD6050(v17 + 1, v18);
          }
        }
      }
LABEL_13:
      v17[4] = v11[4];
      v17[5] = v11[5];
      v11[5] = 0;
      ++*(_DWORD *)(a1 + 16);
      v19 = v11[5];
      if ( v19 )
      {
        v20 = *(unsigned int *)(v19 + 184);
        if ( (_DWORD)v20 )
        {
          v54[0] = 2;
          v25 = *(_QWORD *)(v19 + 168);
          v54[1] = 0;
          v55 = -4096;
          v53 = &unk_49DDAE8;
          v57 = &unk_49DDAE8;
          v56 = 0;
          v58 = 2;
          v26 = v25 + 88 * v20;
          v27 = -4096;
          v59 = 0;
          v60 = -8192;
          i = 0;
          v41 = v19;
          v40 = v11;
          v28 = v25;
          while ( 1 )
          {
            v29 = *(_QWORD *)(v28 + 24);
            if ( v29 != v27 )
            {
              v27 = v60;
              if ( v29 != v60 )
              {
                v30 = *(_QWORD **)(v28 + 40);
                v31 = 4LL * *(unsigned int *)(v28 + 48);
                v32 = &v30[v31];
                if ( v30 != &v30[v31] )
                {
                  do
                  {
                    v33 = *(v32 - 2);
                    v32 -= 4;
                    if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
                      sub_BD60C0(v32);
                  }
                  while ( v30 != v32 );
                  v32 = *(_QWORD **)(v28 + 40);
                }
                if ( v32 != (_QWORD *)(v28 + 56) )
                  _libc_free(v32, v18);
                v27 = *(_QWORD *)(v28 + 24);
              }
            }
            *(_QWORD *)v28 = &unk_49DB368;
            LOBYTE(v18) = v27 != 0;
            if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
              sub_BD60C0((_QWORD *)(v28 + 8));
            v28 += 88;
            if ( v26 == v28 )
              break;
            v27 = v55;
          }
          v19 = v41;
          v11 = v40;
          v57 = &unk_49DB368;
          if ( v60 != 0 && v60 != -4096 && v60 != -8192 )
            sub_BD60C0(&v58);
          v53 = &unk_49DB368;
          if ( v55 != -4096 && v55 != 0 && v55 != -8192 )
            sub_BD60C0(v54);
          v20 = *(unsigned int *)(v41 + 184);
        }
        v21 = 88 * v20;
        sub_C7D6A0(*(_QWORD *)(v19 + 168), 88 * v20, 8);
        v22 = *(_QWORD **)(v19 + 16);
        v23 = &v22[4 * *(unsigned int *)(v19 + 24)];
        if ( v22 != v23 )
        {
          do
          {
            v24 = *(v23 - 2);
            v23 -= 4;
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              sub_BD60C0(v23);
          }
          while ( v22 != v23 );
          v23 = *(_QWORD **)(v19 + 16);
        }
        if ( v23 != (_QWORD *)(v19 + 32) )
          _libc_free(v23, v21);
        j_j___libc_free_0(v19, 200);
      }
      v10 = v11[3];
      while ( 1 )
      {
LABEL_25:
        *v11 = &unk_49DB368;
        if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
          sub_BD60C0(v11 + 1);
        v11 += 6;
        if ( v45 == v11 )
          break;
        v10 = v47;
        v12 = v11[3];
        if ( v12 != v47 )
          goto LABEL_10;
      }
      v49 = &unk_49DB368;
      if ( v51 != 0 && v51 != -8192 && v51 != -4096 )
        sub_BD60C0(v50);
    }
    if ( v47 != 0 && v47 != -4096 && v47 != -8192 )
      sub_BD60C0(v46);
    return (_QWORD *)sub_C7D6A0(v43, v42, 8);
  }
  else
  {
    v58 = 2;
    i = 0;
    v34 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[6 * v34]; j != result; result += 6 )
    {
      if ( result )
      {
        v36 = v58;
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DDB10;
        result[1] = v36 & 6;
        result[4] = i;
      }
    }
  }
  return result;
}

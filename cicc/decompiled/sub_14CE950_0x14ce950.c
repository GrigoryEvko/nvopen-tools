// Function: sub_14CE950
// Address: 0x14ce950
//
_QWORD *__fastcall sub_14CE950(__int64 a1, int a2)
{
  __int64 v2; // r12
  _QWORD *v3; // r15
  unsigned __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rcx
  char v8; // dl
  __int64 v9; // rax
  _QWORD *v10; // r14
  __int64 v11; // rcx
  int v12; // eax
  int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // eax
  _QWORD *v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // rcx
  __int64 v28; // r14
  __int64 v29; // rdi
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // rdx
  _QWORD *j; // rcx
  char v34; // dl
  int v35; // r10d
  _QWORD *v36; // r9
  __int64 v37; // rax
  _QWORD *v38; // [rsp+18h] [rbp-118h]
  __int64 v39; // [rsp+20h] [rbp-110h]
  _QWORD *v40; // [rsp+28h] [rbp-108h]
  _QWORD *v42; // [rsp+38h] [rbp-F8h]
  _QWORD v43[2]; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v44; // [rsp+58h] [rbp-D8h]
  __int64 v45; // [rsp+60h] [rbp-D0h]
  void *v46; // [rsp+70h] [rbp-C0h]
  _QWORD v47[2]; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v48; // [rsp+88h] [rbp-A8h]
  __int64 v49; // [rsp+90h] [rbp-A0h]
  void *v50; // [rsp+A0h] [rbp-90h]
  _QWORD v51[2]; // [rsp+A8h] [rbp-88h] BYREF
  __int64 v52; // [rsp+B8h] [rbp-78h]
  __int64 v53; // [rsp+C0h] [rbp-70h]
  void *v54; // [rsp+D0h] [rbp-60h]
  __int64 v55; // [rsp+D8h] [rbp-58h] BYREF
  __int64 v56; // [rsp+E0h] [rbp-50h]
  __int64 v57; // [rsp+E8h] [rbp-48h]
  __int64 i; // [rsp+F0h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD **)(a1 + 8);
  v40 = v3;
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(48LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v55 = 2;
    v56 = 0;
    v42 = &v3[6 * v2];
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v57 = -8;
    v54 = &unk_49ECBF8;
    v7 = &result[6 * v6];
    for ( i = 0; v7 != result; result += 6 )
    {
      if ( result )
      {
        v8 = v55;
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49ECBF8;
        result[1] = v8 & 6;
        result[4] = i;
      }
    }
    v43[0] = 2;
    v43[1] = 0;
    v44 = -8;
    v45 = 0;
    v47[0] = 2;
    v47[1] = 0;
    v48 = -16;
    v46 = &unk_49ECBF8;
    v49 = 0;
    if ( v42 != v3 )
    {
      v9 = -8;
      v10 = v3;
      v11 = v3[3];
      if ( v11 == -8 )
        goto LABEL_25;
LABEL_10:
      v9 = v48;
      if ( v11 == v48 )
        goto LABEL_25;
      v12 = *(_DWORD *)(a1 + 24);
      if ( !v12 )
        BUG();
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 8);
      v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = (_QWORD *)(v14 + 48LL * v15);
      v17 = v16[3];
      if ( v11 != v17 )
      {
        v35 = 1;
        v36 = 0;
        while ( v17 != -8 )
        {
          if ( !v36 && v17 == -16 )
            v36 = v16;
          v15 = v13 & (v35 + v15);
          v16 = (_QWORD *)(v14 + 48LL * v15);
          v17 = v16[3];
          if ( v11 == v17 )
            goto LABEL_13;
          ++v35;
        }
        if ( v36 )
        {
          v37 = v36[3];
          v16 = v36;
        }
        else
        {
          v37 = v16[3];
        }
        if ( v11 != v37 )
        {
          if ( v37 != -8 && v37 != 0 && v37 != -16 )
          {
            sub_1649B30(v16 + 1);
            v11 = v10[3];
          }
          v16[3] = v11;
          if ( v11 != -8 && v11 != 0 && v11 != -16 )
            sub_1649AC0(v16 + 1, v10[1] & 0xFFFFFFFFFFFFFFF8LL);
        }
      }
LABEL_13:
      v16[4] = v10[4];
      v16[5] = v10[5];
      v10[5] = 0;
      ++*(_DWORD *)(a1 + 16);
      v18 = v10[5];
      if ( v18 )
      {
        v19 = *(unsigned int *)(v18 + 176);
        if ( (_DWORD)v19 )
        {
          v51[0] = 2;
          v23 = *(_QWORD *)(v18 + 160);
          v51[1] = 0;
          v52 = -8;
          v50 = &unk_49ECBD0;
          v54 = &unk_49ECBD0;
          v53 = 0;
          v55 = 2;
          v24 = v23 + 88 * v19;
          v25 = -8;
          v56 = 0;
          v57 = -16;
          i = 0;
          v39 = v18;
          v38 = v10;
          v26 = v23;
          while ( 1 )
          {
            v27 = *(_QWORD *)(v26 + 24);
            if ( v27 != v25 )
            {
              v25 = v57;
              if ( v27 != v57 )
              {
                v28 = *(_QWORD *)(v26 + 40);
                v29 = 32LL * *(unsigned int *)(v26 + 48);
                v30 = v28 + v29;
                if ( v28 != v28 + v29 )
                {
                  do
                  {
                    v31 = *(_QWORD *)(v30 - 16);
                    v30 -= 32LL;
                    if ( v31 != 0 && v31 != -8 && v31 != -16 )
                      sub_1649B30(v30);
                  }
                  while ( v28 != v30 );
                  v30 = *(_QWORD *)(v26 + 40);
                }
                if ( v30 != v26 + 56 )
                  _libc_free(v30);
                v25 = *(_QWORD *)(v26 + 24);
              }
            }
            *(_QWORD *)v26 = &unk_49EE2B0;
            if ( v25 != 0 && v25 != -8 && v25 != -16 )
              sub_1649B30(v26 + 8);
            v26 += 88;
            if ( v24 == v26 )
              break;
            v25 = v52;
          }
          v18 = v39;
          v10 = v38;
          v54 = &unk_49EE2B0;
          if ( v57 != 0 && v57 != -8 && v57 != -16 )
            sub_1649B30(&v55);
          v50 = &unk_49EE2B0;
          if ( v52 != 0 && v52 != -8 && v52 != -16 )
            sub_1649B30(v51);
        }
        j___libc_free_0(*(_QWORD *)(v18 + 160));
        v20 = *(_QWORD *)(v18 + 8);
        v21 = v20 + 32LL * *(unsigned int *)(v18 + 16);
        if ( v20 != v21 )
        {
          do
          {
            v22 = *(_QWORD *)(v21 - 16);
            v21 -= 32LL;
            if ( v22 != 0 && v22 != -8 && v22 != -16 )
              sub_1649B30(v21);
          }
          while ( v20 != v21 );
          v21 = *(_QWORD *)(v18 + 8);
        }
        if ( v21 != v18 + 24 )
          _libc_free(v21);
        j_j___libc_free_0(v18, 192);
      }
      v9 = v10[3];
      while ( 1 )
      {
LABEL_25:
        *v10 = &unk_49EE2B0;
        if ( v9 != -8 && v9 != 0 && v9 != -16 )
          sub_1649B30(v10 + 1);
        v10 += 6;
        if ( v42 == v10 )
          break;
        v9 = v44;
        v11 = v10[3];
        if ( v11 != v44 )
          goto LABEL_10;
      }
      v46 = &unk_49EE2B0;
      if ( v48 != 0 && v48 != -8 && v48 != -16 )
        sub_1649B30(v47);
    }
    if ( v44 != 0 && v44 != -8 && v44 != -16 )
      sub_1649B30(v43);
    return (_QWORD *)j___libc_free_0(v40);
  }
  else
  {
    v55 = 2;
    i = 0;
    v32 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[6 * v32]; j != result; result += 6 )
    {
      if ( result )
      {
        v34 = v55;
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49ECBF8;
        result[1] = v34 & 6;
        result[4] = i;
      }
    }
  }
  return result;
}

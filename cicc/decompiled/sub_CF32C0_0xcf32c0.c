// Function: sub_CF32C0
// Address: 0xcf32c0
//
_QWORD *__fastcall sub_CF32C0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 v8; // rcx
  _QWORD *v9; // r12
  _QWORD *v10; // rcx
  char v11; // dl
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // r13
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rcx
  _QWORD *j; // rcx
  char v25; // dl
  int v26; // r10d
  _QWORD *v27; // r9
  __int64 v28; // rcx
  unsigned __int64 *v29; // r8
  __int64 v30; // [rsp+8h] [rbp-98h]
  _QWORD v31[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v32; // [rsp+28h] [rbp-78h]
  __int64 v33; // [rsp+30h] [rbp-70h]
  void *v34; // [rsp+40h] [rbp-60h]
  _QWORD v35[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v36; // [rsp+58h] [rbp-48h]
  __int64 i; // [rsp+60h] [rbp-40h]

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
  result = (_QWORD *)sub_C7D670((unsigned __int64)v6 << 6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v30 = v4 << 6;
    v9 = (_QWORD *)(v5 + (v4 << 6));
    v35[0] = 2;
    v10 = &result[8 * v8];
    for ( i = 0; v10 != result; result += 8 )
    {
      if ( result )
      {
        v11 = v35[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DD7B0;
        result[1] = v11 & 6;
        result[4] = i;
      }
    }
    v31[1] = 0;
    v12 = (_QWORD *)v5;
    v34 = &unk_49DD7B0;
    v13 = -4096;
    v31[0] = 2;
    v32 = -4096;
    v33 = 0;
    v35[0] = 2;
    v35[1] = 0;
    v36 = -8192;
    i = 0;
    if ( v9 != (_QWORD *)v5 )
    {
      while ( 1 )
      {
        v14 = v12[3];
        if ( v14 != v13 )
        {
          v13 = v36;
          if ( v14 != v36 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
              BUG();
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
            v19 = v18[3];
            if ( v14 != v19 )
            {
              v26 = 1;
              v27 = 0;
              while ( v19 != -4096 )
              {
                if ( v19 == -8192 && !v27 )
                  v27 = v18;
                v17 = (v15 - 1) & (v26 + v17);
                v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
                v19 = v18[3];
                if ( v14 == v19 )
                  goto LABEL_15;
                ++v26;
              }
              if ( v27 )
              {
                v28 = v27[3];
                v18 = v27;
              }
              else
              {
                v28 = v18[3];
              }
              v29 = v18 + 1;
              if ( v14 != v28 )
              {
                if ( v28 != 0 && v28 != -4096 && v28 != -8192 )
                {
                  sub_BD60C0(v18 + 1);
                  v14 = v12[3];
                  v29 = v18 + 1;
                }
                v18[3] = v14;
                if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
                  sub_BD6050(v29, v12[1] & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_15:
            v20 = v12[4];
            v18[5] = 6;
            v18[6] = 0;
            v18[4] = v20;
            v21 = v12[7];
            v18[7] = v21;
            if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
              sub_BD6050(v18 + 5, v12[5] & 0xFFFFFFFFFFFFFFF8LL);
            ++*(_DWORD *)(a1 + 16);
            v22 = v12[7];
            if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
              sub_BD60C0(v12 + 5);
            v13 = v12[3];
          }
        }
        *v12 = &unk_49DB368;
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
          sub_BD60C0(v12 + 1);
        v12 += 8;
        if ( v9 == v12 )
          break;
        v13 = v32;
      }
      v34 = &unk_49DB368;
      if ( v36 != 0 && v36 != -4096 && v36 != -8192 )
        sub_BD60C0(v35);
    }
    if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
      sub_BD60C0(v31);
    return (_QWORD *)sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v35[0] = 2;
    i = 0;
    for ( j = &result[8 * v23]; j != result; result += 8 )
    {
      if ( result )
      {
        v25 = v35[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_49DD7B0;
        result[1] = v25 & 6;
        result[4] = i;
      }
    }
  }
  return result;
}

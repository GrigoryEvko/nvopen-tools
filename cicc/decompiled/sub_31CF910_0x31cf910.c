// Function: sub_31CF910
// Address: 0x31cf910
//
_QWORD *__fastcall sub_31CF910(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r12
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r13
  _QWORD *i; // rsi
  char v11; // dl
  _QWORD *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // eax
  int v16; // esi
  __int64 v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // r15
  __int64 v20; // r8
  __int64 v21; // rdx
  _QWORD *j; // rcx
  char v23; // dl
  int v24; // r10d
  _QWORD *v25; // r9
  __int64 v26; // rax
  unsigned __int64 *v27; // r8
  __int64 v28; // [rsp+8h] [rbp-98h]
  _QWORD v29[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v30; // [rsp+28h] [rbp-78h]
  __int64 v31; // [rsp+30h] [rbp-70h]
  void *v32; // [rsp+40h] [rbp-60h]
  _QWORD v33[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v34; // [rsp+58h] [rbp-48h]
  __int64 v35; // [rsp+60h] [rbp-40h]

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
  result = (_QWORD *)sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v33[0] = 2;
    v28 = 48 * v4;
    v9 = (_QWORD *)(v5 + 48 * v4);
    v35 = 0;
    for ( i = &result[6 * v8]; i != result; result += 6 )
    {
      if ( result )
      {
        v11 = v33[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_4A34DD0;
        result[1] = v11 & 6;
        result[4] = v35;
      }
    }
    v29[1] = 0;
    v12 = (_QWORD *)v5;
    v32 = &unk_4A34DD0;
    v13 = -4096;
    v29[0] = 2;
    v30 = -4096;
    v31 = 0;
    v33[0] = 2;
    v33[1] = 0;
    v34 = -8192;
    v35 = 0;
    if ( v9 != (_QWORD *)v5 )
    {
      while ( 1 )
      {
        v14 = v12[3];
        if ( v14 != v13 )
        {
          v13 = v34;
          if ( v14 != v34 )
          {
            v15 = *(_DWORD *)(a1 + 24);
            if ( !v15 )
              BUG();
            v16 = v15 - 1;
            v17 = *(_QWORD *)(a1 + 8);
            v18 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v19 = (_QWORD *)(v17 + 48LL * v18);
            v20 = v19[3];
            if ( v14 != v20 )
            {
              v24 = 1;
              v25 = 0;
              while ( v20 != -4096 )
              {
                if ( v20 == -8192 && !v25 )
                  v25 = v19;
                v18 = v16 & (v24 + v18);
                v19 = (_QWORD *)(v17 + 48LL * v18);
                v20 = v19[3];
                if ( v14 == v20 )
                  goto LABEL_15;
                ++v24;
              }
              if ( v25 )
              {
                v26 = v25[3];
                v19 = v25;
              }
              else
              {
                v26 = v19[3];
              }
              v27 = v19 + 1;
              if ( v14 != v26 )
              {
                if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
                {
                  sub_BD60C0(v19 + 1);
                  v14 = v12[3];
                  v27 = v19 + 1;
                }
                v19[3] = v14;
                if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
                  sub_BD6050(v27, v12[1] & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_15:
            v19[4] = v12[4];
            v19[5] = v12[5];
            ++*(_DWORD *)(a1 + 16);
            v13 = v12[3];
          }
        }
        *v12 = &unk_49DB368;
        if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
          sub_BD60C0(v12 + 1);
        v12 += 6;
        if ( v9 == v12 )
          break;
        v13 = v30;
      }
      v32 = &unk_49DB368;
      if ( v34 != -4096 && v34 != 0 && v34 != -8192 )
        sub_BD60C0(v33);
    }
    if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
      sub_BD60C0(v29);
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v33[0] = 2;
    v35 = 0;
    for ( j = &result[6 * v21]; j != result; result += 6 )
    {
      if ( result )
      {
        v23 = v33[0];
        result[2] = 0;
        result[3] = -4096;
        *result = &unk_4A34DD0;
        result[1] = v23 & 6;
        result[4] = v35;
      }
    }
  }
  return result;
}

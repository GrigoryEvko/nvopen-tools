// Function: sub_12E48B0
// Address: 0x12e48b0
//
_QWORD *__fastcall sub_12E48B0(__int64 a1, int a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  _QWORD *v8; // r12
  _QWORD *i; // rcx
  char v10; // dl
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rsi
  unsigned int v17; // eax
  _QWORD *v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rcx
  _QWORD *j; // rcx
  char v25; // dl
  int v26; // r9d
  _QWORD *v27; // r8
  __int64 v28; // rax
  _QWORD *v29; // rdi
  _QWORD v30[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v31; // [rsp+28h] [rbp-78h]
  __int64 v32; // [rsp+30h] [rbp-70h]
  void *v33; // [rsp+40h] [rbp-60h]
  _QWORD v34[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v35; // [rsp+58h] [rbp-48h]
  __int64 v36; // [rsp+60h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
  v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v34[0] = 2;
    v8 = &v4[8 * v3];
    v36 = 0;
    for ( i = &result[8 * v7]; i != result; result += 8 )
    {
      if ( result )
      {
        v10 = v34[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49E6B50;
        result[1] = v10 & 6;
        result[4] = v36;
      }
    }
    v30[1] = 0;
    v11 = v4;
    v33 = &unk_49E6B50;
    v12 = -8;
    v30[0] = 2;
    v31 = -8;
    v32 = 0;
    v34[0] = 2;
    v34[1] = 0;
    v35 = -16;
    v36 = 0;
    if ( v8 != v4 )
    {
      while ( 1 )
      {
        v13 = v11[3];
        if ( v13 != v12 )
        {
          v12 = v35;
          if ( v13 != v35 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
            v19 = v18[3];
            if ( v13 != v19 )
            {
              v26 = 1;
              v27 = 0;
              while ( v19 != -8 )
              {
                if ( v19 == -16 && !v27 )
                  v27 = v18;
                v17 = v15 & (v26 + v17);
                v18 = (_QWORD *)(v16 + ((unsigned __int64)v17 << 6));
                v19 = v18[3];
                if ( v13 == v19 )
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
              if ( v13 != v28 )
              {
                if ( v28 != -8 && v28 != 0 && v28 != -16 )
                {
                  sub_1649B30(v29);
                  v13 = v11[3];
                  v29 = v18 + 1;
                }
                v18[3] = v13;
                if ( v13 != -8 && v13 != 0 && v13 != -16 )
                  sub_1649AC0(v29, v11[1] & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_15:
            v20 = v11[4];
            v18[5] = 6;
            v18[6] = 0;
            v18[4] = v20;
            v21 = v11[7];
            v18[7] = v21;
            if ( v21 != -8 && v21 != 0 && v21 != -16 )
              sub_1649AC0(v18 + 5, v11[5] & 0xFFFFFFFFFFFFFFF8LL);
            ++*(_DWORD *)(a1 + 16);
            v22 = v11[7];
            if ( v22 != 0 && v22 != -8 && v22 != -16 )
              sub_1649B30(v11 + 5);
            v12 = v11[3];
          }
        }
        *v11 = &unk_49EE2B0;
        if ( v12 != 0 && v12 != -8 && v12 != -16 )
          sub_1649B30(v11 + 1);
        v11 += 8;
        if ( v8 == v11 )
          break;
        v12 = v31;
      }
      v33 = &unk_49EE2B0;
      if ( v35 != -8 && v35 != 0 && v35 != -16 )
        sub_1649B30(v34);
    }
    if ( v31 != 0 && v31 != -8 && v31 != -16 )
      sub_1649B30(v30);
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v23 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v34[0] = 2;
    v36 = 0;
    for ( j = &result[8 * v23]; j != result; result += 8 )
    {
      if ( result )
      {
        v25 = v34[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49E6B50;
        result[1] = v25 & 6;
        result[4] = v36;
      }
    }
  }
  return result;
}

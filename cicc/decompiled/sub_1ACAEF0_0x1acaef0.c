// Function: sub_1ACAEF0
// Address: 0x1acaef0
//
__int64 __fastcall sub_1ACAEF0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  unsigned __int8 v4; // al
  unsigned __int64 v5; // rsi
  unsigned __int8 v6; // al
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r14
  unsigned __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  __int64 v21; // r14
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  size_t *v25; // rcx
  unsigned __int64 v26; // r14
  __int64 v27; // rax
  size_t **v28; // rax
  size_t v29; // r9
  const void *v30; // rdi
  size_t v31; // rcx
  const void *v32; // rsi
  unsigned __int64 v33; // r8
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  size_t v38; // [rsp+0h] [rbp-60h]
  unsigned __int64 v39; // [rsp+8h] [rbp-58h]
  size_t *v40; // [rsp+10h] [rbp-50h]
  size_t v41; // [rsp+10h] [rbp-50h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  unsigned __int64 v43; // [rsp+20h] [rbp-40h]
  size_t v45; // [rsp+20h] [rbp-40h]
  size_t v46; // [rsp+28h] [rbp-38h]

  v3 = 0;
  v4 = *(_BYTE *)(a2 + 16);
  if ( v4 > 0x17u )
  {
    v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v4 != 78 )
    {
      v5 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      v3 = 0;
      if ( v4 == 29 )
        v3 = v5;
    }
  }
  v6 = *(_BYTE *)(a3 + 16);
  if ( v6 <= 0x17u )
  {
    v7 = 0;
    goto LABEL_8;
  }
  if ( v6 == 78 )
  {
    v35 = a3 | 4;
  }
  else
  {
    v7 = 0;
    if ( v6 != 29 )
    {
LABEL_8:
      v8 = 0;
      if ( *(char *)(v7 + 23) >= 0 )
        goto LABEL_11;
      v9 = sub_1648A40(v7);
      v11 = v9 + v10;
      if ( *(char *)(v7 + 23) < 0 )
        goto LABEL_10;
      v8 = (unsigned int)(v11 >> 4);
      goto LABEL_11;
    }
    v35 = a3 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v7 = v35 & 0xFFFFFFFFFFFFFFF8LL;
  if ( ((v35 >> 2) & 1) == 0 )
    goto LABEL_8;
  v8 = 0;
  if ( *(char *)((v35 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
  {
    v36 = sub_1648A40(v7);
    v11 = v36 + v37;
    if ( *(char *)(v7 + 23) < 0 )
    {
LABEL_10:
      v8 = (unsigned int)((v11 - sub_1648A40(v7)) >> 4);
      goto LABEL_11;
    }
    v8 = (unsigned int)(v11 >> 4);
  }
LABEL_11:
  v12 = 0;
  if ( *(char *)(v3 + 23) < 0 )
  {
    v43 = v8;
    v13 = sub_1648A40(v3);
    v8 = v43;
    v15 = v13 + v14;
    if ( *(char *)(v3 + 23) >= 0 )
    {
      v12 = (unsigned int)(v15 >> 4);
    }
    else
    {
      v16 = sub_1648A40(v3);
      v8 = v43;
      v12 = (unsigned int)((v15 - v16) >> 4);
    }
  }
  result = sub_1ACA9E0(a1, v12, v8);
  if ( !(_DWORD)result )
  {
    if ( *(char *)(v3 + 23) < 0 )
    {
      v18 = sub_1648A40(v3);
      v20 = v18 + v19;
      if ( *(char *)(v3 + 23) < 0 )
        v20 -= sub_1648A40(v3);
      v21 = v20 >> 4;
      if ( (_DWORD)v21 )
      {
        v22 = 0;
        v42 = 16LL * (unsigned int)v21;
        while ( 1 )
        {
          v23 = 0;
          if ( *(char *)(v3 + 23) < 0 )
            v23 = sub_1648A40(v3);
          v24 = v22 + v23;
          v25 = *(size_t **)v24;
          v26 = 0xAAAAAAAAAAAAAAABLL * (3LL * *(unsigned int *)(v24 + 12) - 3LL * *(unsigned int *)(v24 + 8));
          v27 = 0;
          if ( *(char *)(v7 + 23) < 0 )
          {
            v40 = v25;
            v27 = sub_1648A40(v7);
            v25 = v40;
          }
          v28 = (size_t **)(v22 + v27);
          v29 = *v25;
          v30 = v25 + 2;
          v31 = **v28;
          v32 = *v28 + 2;
          v33 = 0xAAAAAAAAAAAAAAABLL * (3LL * *((unsigned int *)v28 + 3) - 3LL * *((unsigned int *)v28 + 2));
          if ( v31 < v29 )
            break;
          if ( v29 )
          {
            v38 = **v28;
            v39 = 0xAAAAAAAAAAAAAAABLL * (3LL * *((unsigned int *)v28 + 3) - 3LL * *((unsigned int *)v28 + 2));
            v41 = v29;
            v34 = memcmp(v30, v32, v29);
            v29 = v41;
            v33 = v39;
            v31 = v38;
            if ( v34 )
              goto LABEL_38;
          }
          if ( v31 != v29 )
            goto LABEL_28;
          result = sub_1ACA9E0(a1, v26, v33);
          if ( (_DWORD)result )
            return result;
          v22 += 16;
          if ( v42 == v22 )
            return 0;
        }
        v46 = v29;
        if ( !v31 )
          return 1;
        v45 = **v28;
        v34 = memcmp(v30, v32, v31);
        v31 = v45;
        v29 = v46;
        if ( v34 )
        {
LABEL_38:
          if ( v34 >= 0 )
            return 1;
        }
        else
        {
LABEL_28:
          if ( v31 <= v29 )
            return 1;
        }
        return 0xFFFFFFFFLL;
      }
    }
    return 0;
  }
  return result;
}

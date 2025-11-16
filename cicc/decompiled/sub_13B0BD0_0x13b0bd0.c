// Function: sub_13B0BD0
// Address: 0x13b0bd0
//
__int64 __fastcall sub_13B0BD0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  unsigned __int64 *v6; // r15
  unsigned __int64 *v7; // r15
  unsigned __int64 *v8; // r15
  __int64 result; // rax
  __int64 v10; // rbx
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // r13
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  size_t v17; // r9
  size_t v18; // rdx
  void *v19; // r8
  const void *v20; // rsi
  __int64 v21; // r15
  __int64 v22; // rax
  size_t v23; // r9
  __int64 v24; // rdx
  void *v25; // r8
  const void *v26; // rsi
  __int64 v27; // r15
  __int64 v28; // rax
  size_t v29; // r9
  __int64 v30; // rdx
  void *v31; // r8
  const void *v32; // rsi
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rax
  size_t v36; // [rsp+8h] [rbp-48h]
  size_t v37; // [rsp+8h] [rbp-48h]
  void *v38; // [rsp+8h] [rbp-48h]
  size_t v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+10h] [rbp-40h]
  __int64 v41; // [rsp+10h] [rbp-40h]
  __int64 v42; // [rsp+10h] [rbp-40h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  size_t v44; // [rsp+10h] [rbp-40h]
  size_t n; // [rsp+18h] [rbp-38h]
  size_t na; // [rsp+18h] [rbp-38h]
  size_t nb; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)a1 + 48LL * *(unsigned int *)(a1 + 8);
  while ( v4 != v5 )
  {
    while ( 1 )
    {
      v6 = *(unsigned __int64 **)(v5 - 8);
      v5 -= 48;
      if ( ((unsigned __int8)v6 & 1) == 0 && v6 )
      {
        _libc_free(*v6);
        j_j___libc_free_0(v6, 24);
      }
      v7 = *(unsigned __int64 **)(v5 + 32);
      if ( ((unsigned __int8)v7 & 1) == 0 && v7 )
      {
        _libc_free(*v7);
        j_j___libc_free_0(v7, 24);
      }
      v8 = *(unsigned __int64 **)(v5 + 24);
      if ( ((unsigned __int8)v8 & 1) != 0 || !v8 )
        break;
      _libc_free(*v8);
      j_j___libc_free_0(v8, 24);
      if ( v4 == v5 )
        goto LABEL_12;
    }
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = 0;
  if ( a2 > *(unsigned int *)(a1 + 12) )
    sub_13AE5E0(a1, a2);
  result = a2;
  v10 = *(_QWORD *)a1;
  *(_DWORD *)(a1 + 8) = a2;
  v11 = v10 + 48LL * (unsigned int)a2;
  if ( v10 != v11 )
  {
    while ( 1 )
    {
      if ( !v10 )
        goto LABEL_21;
      *(_QWORD *)v10 = *(_QWORD *)a3;
      *(_QWORD *)(v10 + 8) = *(_QWORD *)(a3 + 8);
      result = *(unsigned int *)(a3 + 16);
      *(_QWORD *)(v10 + 24) = 1;
      *(_DWORD *)(v10 + 16) = result;
      v14 = *(_QWORD *)(a3 + 24);
      if ( (v14 & 1) != 0 )
      {
        *(_QWORD *)(v10 + 24) = v14;
      }
      else
      {
        result = sub_22077B0(24);
        v15 = result;
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
          result = *(unsigned int *)(v14 + 16);
          *(_DWORD *)(v15 + 16) = result;
          if ( (_DWORD)result )
          {
            v39 = (unsigned int)(result + 63) >> 6;
            v16 = malloc(8 * v39);
            v17 = 8 * v39;
            v18 = v39;
            v19 = (void *)v16;
            if ( !v16 )
            {
              if ( 8 * v39 || (v35 = malloc(1u), v18 = v39, v17 = 0, v19 = 0, !v35) )
              {
                v38 = v19;
                v44 = v17;
                nb = v18;
                sub_16BD1C0("Allocation failed");
                v18 = nb;
                v17 = v44;
                v19 = v38;
              }
              else
              {
                v19 = (void *)v35;
              }
            }
            *(_QWORD *)(v15 + 8) = v18;
            v20 = *(const void **)v14;
            *(_QWORD *)v15 = v19;
            result = (__int64)memcpy(v19, v20, v17);
          }
        }
        *(_QWORD *)(v10 + 24) = v15;
      }
      *(_QWORD *)(v10 + 32) = 1;
      v12 = *(_QWORD *)(a3 + 32);
      if ( (v12 & 1) != 0 )
      {
        *(_QWORD *)(v10 + 32) = v12;
      }
      else
      {
        result = sub_22077B0(24);
        v27 = result;
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
          result = *(unsigned int *)(v12 + 16);
          *(_DWORD *)(v27 + 16) = result;
          if ( (_DWORD)result )
          {
            v41 = (unsigned int)(result + 63) >> 6;
            v28 = malloc(8 * v41);
            v29 = 8 * v41;
            v30 = v41;
            v31 = (void *)v28;
            if ( !v28 )
            {
              if ( 8 * v41 || (v33 = malloc(1u), v31 = 0, v30 = v41, v29 = 0, !v33) )
              {
                v36 = v29;
                v42 = v30;
                n = (size_t)v31;
                sub_16BD1C0("Allocation failed");
                v31 = (void *)n;
                v30 = v42;
                v29 = v36;
              }
              else
              {
                v31 = (void *)v33;
              }
            }
            *(_QWORD *)(v27 + 8) = v30;
            v32 = *(const void **)v12;
            *(_QWORD *)v27 = v31;
            result = (__int64)memcpy(v31, v32, v29);
          }
        }
        *(_QWORD *)(v10 + 32) = v27;
      }
      *(_QWORD *)(v10 + 40) = 1;
      v13 = *(_QWORD *)(a3 + 40);
      if ( (v13 & 1) != 0 )
      {
        *(_QWORD *)(v10 + 40) = v13;
LABEL_21:
        v10 += 48;
        if ( v11 == v10 )
          return result;
      }
      else
      {
        result = sub_22077B0(24);
        v21 = result;
        if ( result )
        {
          *(_QWORD *)result = 0;
          *(_QWORD *)(result + 8) = 0;
          result = *(unsigned int *)(v13 + 16);
          *(_DWORD *)(v21 + 16) = result;
          if ( (_DWORD)result )
          {
            v40 = (unsigned int)(result + 63) >> 6;
            v22 = malloc(8 * v40);
            v23 = 8 * v40;
            v24 = v40;
            v25 = (void *)v22;
            if ( !v22 )
            {
              if ( 8 * v40 || (v34 = malloc(1u), v25 = 0, v24 = v40, v23 = 0, !v34) )
              {
                v37 = v23;
                v43 = v24;
                na = (size_t)v25;
                sub_16BD1C0("Allocation failed");
                v25 = (void *)na;
                v24 = v43;
                v23 = v37;
              }
              else
              {
                v25 = (void *)v34;
              }
            }
            *(_QWORD *)(v21 + 8) = v24;
            v26 = *(const void **)v13;
            *(_QWORD *)v21 = v25;
            result = (__int64)memcpy(v25, v26, v23);
          }
        }
        *(_QWORD *)(v10 + 40) = v21;
        v10 += 48;
        if ( v11 == v10 )
          return result;
      }
    }
  }
  return result;
}

// Function: sub_12BDBC0
// Address: 0x12bdbc0
//
unsigned __int64 __fastcall sub_12BDBC0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v7; // rdi
  size_t v8; // rdx
  size_t v9; // rcx
  int v10; // r8d
  unsigned int v11; // eax
  const void *v12; // rsi
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  int v15; // r8d
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rax
  int v21; // r15d
  __int64 v22; // r14
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v26; // rdx
  int v27; // r14d
  __int64 v28; // r13
  __int64 v29; // rax
  __int64 v30; // rdx
  const void *v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // [rsp+0h] [rbp-40h]
  int v36; // [rsp+0h] [rbp-40h]
  int v37; // [rsp+8h] [rbp-38h]
  int v38; // [rsp+8h] [rbp-38h]
  int v39; // [rsp+8h] [rbp-38h]
  int v40; // [rsp+8h] [rbp-38h]
  int v41; // [rsp+8h] [rbp-38h]
  int v42; // [rsp+8h] [rbp-38h]
  int v43; // [rsp+8h] [rbp-38h]
  size_t v44; // [rsp+8h] [rbp-38h]

  v2 = a1 + 64;
  v3 = a1 + 240;
  v4 = a1 + 424;
  *(_QWORD *)(a1 + 32) = 0x800000000LL;
  *(_QWORD *)(a1 + 232) = 0x800000000LL;
  *(_QWORD *)(a1 + 416) = 0x800000000LL;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 56) = 0x1000000000LL;
  *(_QWORD *)(a1 + 192) = a1 + 208;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 200) = 0;
  *(_BYTE *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 224) = a1 + 240;
  *(_QWORD *)(a1 + 400) = 0;
  *(_QWORD *)(a1 + 408) = a1 + 424;
  v35 = a1 + 40;
  sub_15A9210();
  sub_2240AE0(a1 + 192, a2 + 192);
  v7 = a1 + 24;
  *(_BYTE *)a1 = *(_BYTE *)a2;
  *(_QWORD *)(a1 + 4) = *(_QWORD *)(a2 + 4);
  *(_QWORD *)(a1 + 12) = *(_QWORD *)(a2 + 12);
  if ( a1 + 24 != a2 + 24 )
  {
    v8 = *(unsigned int *)(a2 + 32);
    v9 = *(unsigned int *)(a1 + 32);
    v10 = *(_DWORD *)(a2 + 32);
    if ( v8 <= v9 )
    {
      if ( *(_DWORD *)(a2 + 32) )
      {
        v40 = *(_DWORD *)(a2 + 32);
        memmove(*(void **)(a1 + 24), *(const void **)(a2 + 24), v8);
        v10 = v40;
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 36) )
      {
        *(_DWORD *)(a1 + 32) = 0;
        v42 = v8;
        sub_16CD150(v7, v35, v8, 1);
        v8 = *(unsigned int *)(a2 + 32);
        v10 = v42;
        v9 = 0;
        v11 = *(_DWORD *)(a2 + 32);
      }
      else
      {
        v11 = *(_DWORD *)(a2 + 32);
        if ( *(_DWORD *)(a1 + 32) )
        {
          v36 = *(_DWORD *)(a2 + 32);
          v44 = *(unsigned int *)(a1 + 32);
          memmove(*(void **)(a1 + 24), *(const void **)(a2 + 24), v9);
          v8 = *(unsigned int *)(a2 + 32);
          v10 = v36;
          v9 = v44;
          v11 = *(_DWORD *)(a2 + 32);
        }
      }
      v12 = (const void *)(v9 + *(_QWORD *)(a2 + 24));
      if ( v12 != (const void *)(*(_QWORD *)(a2 + 24) + v8) )
      {
        v37 = v10;
        memcpy((void *)(v9 + *(_QWORD *)(a1 + 24)), v12, v11 - v9);
        v10 = v37;
      }
    }
    *(_DWORD *)(a1 + 32) = v10;
  }
  if ( a1 + 48 != a2 + 48 )
  {
    v13 = *(unsigned int *)(a2 + 56);
    v14 = *(unsigned int *)(a1 + 56);
    v15 = *(_DWORD *)(a2 + 56);
    if ( v13 <= v14 )
    {
      if ( *(_DWORD *)(a2 + 56) )
      {
        v39 = *(_DWORD *)(a2 + 56);
        memmove(*(void **)(a1 + 48), *(const void **)(a2 + 48), 8 * v13);
        v15 = v39;
      }
    }
    else
    {
      if ( v13 > *(unsigned int *)(a1 + 60) )
      {
        *(_DWORD *)(a1 + 56) = 0;
        v32 = v2;
        v16 = 0;
        v41 = v13;
        sub_16CD150(a1 + 48, v32, v13, 8);
        v13 = *(unsigned int *)(a2 + 56);
        v15 = v41;
      }
      else
      {
        v16 = 8 * v14;
        if ( *(_DWORD *)(a1 + 56) )
        {
          v43 = *(_DWORD *)(a2 + 56);
          memmove(*(void **)(a1 + 48), *(const void **)(a2 + 48), 8 * v14);
          v13 = *(unsigned int *)(a2 + 56);
          v15 = v43;
        }
      }
      v17 = *(_QWORD *)(a2 + 48);
      v18 = 8 * v13;
      if ( v17 + v16 != v18 + v17 )
      {
        v38 = v15;
        memcpy((void *)(v16 + *(_QWORD *)(a1 + 48)), (const void *)(v17 + v16), v18 - v16);
        v15 = v38;
      }
    }
    *(_DWORD *)(a1 + 56) = v15;
  }
  if ( a1 + 224 != a2 + 224 )
  {
    v19 = *(unsigned int *)(a2 + 232);
    v20 = *(unsigned int *)(a1 + 232);
    v21 = *(_DWORD *)(a2 + 232);
    if ( v19 <= v20 )
    {
      if ( *(_DWORD *)(a2 + 232) )
        memmove(*(void **)(a1 + 224), *(const void **)(a2 + 224), 20 * v19);
    }
    else
    {
      if ( v19 > *(unsigned int *)(a1 + 236) )
      {
        *(_DWORD *)(a1 + 232) = 0;
        v33 = v3;
        v22 = 0;
        sub_16CD150(a1 + 224, v33, v19, 20);
        v19 = *(unsigned int *)(a2 + 232);
      }
      else
      {
        v22 = 20 * v20;
        if ( *(_DWORD *)(a1 + 232) )
        {
          memmove(*(void **)(a1 + 224), *(const void **)(a2 + 224), 20 * v20);
          v19 = *(unsigned int *)(a2 + 232);
        }
      }
      v23 = *(_QWORD *)(a2 + 224);
      v24 = 20 * v19;
      if ( v23 + v22 != v24 + v23 )
        memcpy((void *)(v22 + *(_QWORD *)(a1 + 224)), (const void *)(v23 + v22), v24 - v22);
    }
    *(_DWORD *)(a1 + 232) = v21;
  }
  result = a2 + 408;
  if ( a1 + 408 != a2 + 408 )
  {
    v26 = *(unsigned int *)(a2 + 416);
    result = *(unsigned int *)(a1 + 416);
    v27 = *(_DWORD *)(a2 + 416);
    if ( v26 <= result )
    {
      if ( *(_DWORD *)(a2 + 416) )
        result = (unsigned __int64)memmove(*(void **)(a1 + 408), *(const void **)(a2 + 408), 4 * v26);
    }
    else
    {
      if ( v26 > *(unsigned int *)(a1 + 420) )
      {
        *(_DWORD *)(a1 + 416) = 0;
        v34 = v4;
        v28 = 0;
        sub_16CD150(a1 + 408, v34, v26, 4);
        v26 = *(unsigned int *)(a2 + 416);
      }
      else
      {
        v28 = 4 * result;
        if ( *(_DWORD *)(a1 + 416) )
        {
          memmove(*(void **)(a1 + 408), *(const void **)(a2 + 408), 4 * result);
          v26 = *(unsigned int *)(a2 + 416);
        }
      }
      v29 = *(_QWORD *)(a2 + 408);
      v30 = 4 * v26;
      v31 = (const void *)(v29 + v28);
      result = v30 + v29;
      if ( v31 != (const void *)result )
        result = (unsigned __int64)memcpy((void *)(v28 + *(_QWORD *)(a1 + 408)), v31, v30 - v28);
    }
    *(_DWORD *)(a1 + 416) = v27;
  }
  return result;
}

// Function: sub_34B3820
// Address: 0x34b3820
//
void __fastcall sub_34B3820(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 (*v8)(void); // rdx
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // rax
  _QWORD **v15; // r12
  _QWORD **v16; // r14
  unsigned __int64 v17; // rdx
  unsigned int v18; // r10d
  char *v19; // rdi
  char *v20; // rsi
  __int64 v21; // rdx
  size_t v22; // r15
  _QWORD *v23; // rax
  _QWORD *v24; // rcx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rdx
  __int64 v29; // rcx
  int v30; // ecx
  int v31; // eax
  unsigned __int64 v32; // r10
  const void *v33; // [rsp+0h] [rbp-A0h]
  unsigned int v34; // [rsp+8h] [rbp-98h]
  unsigned int v35; // [rsp+8h] [rbp-98h]
  unsigned int v36; // [rsp+8h] [rbp-98h]
  unsigned __int64 v37; // [rsp+8h] [rbp-98h]
  unsigned int v38; // [rsp+8h] [rbp-98h]
  unsigned __int64 v39; // [rsp+8h] [rbp-98h]
  void **v40; // [rsp+10h] [rbp-90h]
  void *src; // [rsp+20h] [rbp-80h] BYREF
  unsigned int v42; // [rsp+28h] [rbp-78h]
  char v43; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v44; // [rsp+60h] [rbp-40h]

  *(_QWORD *)a1 = off_49D8CF0;
  v7 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a1 + 8) = a2;
  *(_QWORD *)(a1 + 16) = v7;
  v8 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  v9 = 0;
  if ( v8 != sub_2DAC790 )
  {
    v9 = v8();
    a2 = *(_QWORD *)(a1 + 8);
  }
  *(_QWORD *)(a1 + 24) = v9;
  v10 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 32) = v10;
  v13 = v10;
  v40 = (void **)(a1 + 48);
  *(_QWORD *)(a1 + 48) = a1 + 64;
  v33 = (const void *)(a1 + 64);
  *(_QWORD *)(a1 + 56) = 0x600000000LL;
  v14 = *(unsigned int *)(a4 + 8);
  *(_QWORD *)(a1 + 40) = a3;
  v15 = *(_QWORD ***)a4;
  *(_QWORD *)(a1 + 120) = 0;
  v16 = &v15[v14];
  if ( v15 != v16 )
  {
    while ( 1 )
    {
      sub_2FF67E0((__int64)&src, v13, *(_QWORD *)(a1 + 8), *v15, v11, v12);
      v22 = 8LL * *(unsigned int *)(a1 + 56);
      v23 = sub_34B2960(*(_QWORD **)(a1 + 48), *(_QWORD *)(a1 + 48) + v22);
      if ( v24 == v23 )
        break;
      v25 = v44;
      if ( *(_DWORD *)(a1 + 112) < v44 )
      {
        v30 = *(_DWORD *)(a1 + 112) & 0x3F;
        if ( v30 )
        {
          *(_QWORD *)(v11 + v22 - 8) &= ~(-1LL << v30);
          v12 = *(unsigned int *)(a1 + 56);
        }
        *(_DWORD *)(a1 + 112) = v25;
        v11 = (v25 + 63) >> 6;
        if ( v11 != v12 )
        {
          if ( v11 >= v12 )
          {
            v32 = v11 - v12;
            if ( v11 > *(unsigned int *)(a1 + 60) )
            {
              v39 = v11 - v12;
              sub_C8D5F0((__int64)v40, v33, v11, 8u, v11, v12);
              v12 = *(unsigned int *)(a1 + 56);
              v32 = v39;
            }
            if ( 8 * v32 )
            {
              v37 = v32;
              memset((void *)(*(_QWORD *)(a1 + 48) + 8 * v12), 0, 8 * v32);
              v12 = *(unsigned int *)(a1 + 56);
              v32 = v37;
            }
            v12 += v32;
            v25 = *(_DWORD *)(a1 + 112);
            *(_DWORD *)(a1 + 56) = v12;
          }
          else
          {
            *(_DWORD *)(a1 + 56) = (v25 + 63) >> 6;
          }
        }
        v31 = v25 & 0x3F;
        if ( v31 )
          *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL * *(unsigned int *)(a1 + 56) - 8) &= ~(-1LL << v31);
      }
      v26 = 0;
      v27 = 8LL * v42;
      if ( v42 )
      {
        do
        {
          v28 = (_QWORD *)(v26 + *(_QWORD *)(a1 + 48));
          v29 = *(_QWORD *)((char *)src + v26);
          v26 += 8;
          *v28 |= v29;
        }
        while ( v26 != v27 );
      }
      v19 = (char *)src;
LABEL_13:
      if ( v19 != &v43 )
        _libc_free((unsigned __int64)v19);
      if ( v16 == ++v15 )
        return;
      v13 = *(_QWORD *)(a1 + 32);
    }
    if ( v40 != &src )
    {
      v17 = v42;
      v18 = v42;
      if ( v12 < v42 )
      {
        if ( v42 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
        {
          *(_DWORD *)(a1 + 56) = 0;
          v36 = v17;
          sub_C8D5F0((__int64)v40, v33, v17, 8u, v11, v12);
          v19 = (char *)src;
          v11 = *(_QWORD *)(a1 + 48);
          v12 = 0;
          v17 = v42;
          v18 = v36;
          v20 = (char *)src;
        }
        else
        {
          v19 = (char *)src;
          v20 = (char *)src;
          if ( v12 )
          {
            if ( v22 )
            {
              v38 = v42;
              memmove((void *)v11, src, v22);
              v19 = (char *)src;
              v12 = v22;
              v17 = v42;
              v18 = v38;
              v11 = v22 + *(_QWORD *)(a1 + 48);
              v20 = (char *)src + v22;
            }
            else
            {
              v11 = (unsigned __int64)v24;
              v12 = 0;
            }
          }
        }
        v21 = 8 * v17;
        if ( v20 != &v19[v21] )
        {
          v34 = v18;
          memcpy((void *)v11, v20, v21 - v12);
          v19 = (char *)src;
          v18 = v34;
        }
        *(_DWORD *)(a1 + 56) = v18;
        goto LABEL_12;
      }
      if ( v42 )
      {
        v35 = v42;
        memmove((void *)v11, src, 8LL * v42);
        v18 = v35;
      }
      *(_DWORD *)(a1 + 56) = v18;
    }
    v19 = (char *)src;
LABEL_12:
    *(_DWORD *)(a1 + 112) = v44;
    goto LABEL_13;
  }
}

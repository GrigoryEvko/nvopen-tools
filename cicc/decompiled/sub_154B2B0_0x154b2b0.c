// Function: sub_154B2B0
// Address: 0x154b2b0
//
void __fastcall sub_154B2B0(__int64 *a1, const char *a2, size_t a3, unsigned int a4)
{
  __int64 v7; // r12
  _WORD *v8; // rdi
  unsigned __int64 v9; // rax
  unsigned int v10; // eax
  unsigned int *v11; // rdi
  unsigned int *v12; // r13
  unsigned int *v13; // r14
  char v14; // r12
  void *v15; // rdi
  __int64 v16; // rax
  __int64 v17; // r15
  const char *v18; // r10
  size_t v19; // rdx
  size_t v20; // r8
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // rax
  size_t v27; // [rsp-80h] [rbp-80h]
  unsigned int v28; // [rsp-74h] [rbp-74h]
  size_t v29; // [rsp-70h] [rbp-70h]
  const char *v30; // [rsp-70h] [rbp-70h]
  unsigned int *v31; // [rsp-68h] [rbp-68h] BYREF
  __int64 v32; // [rsp-60h] [rbp-60h]
  _BYTE v33[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( a4 )
  {
    v7 = *a1;
    if ( *((_BYTE *)a1 + 8) )
      *((_BYTE *)a1 + 8) = 0;
    else
      v7 = sub_1263B40(*a1, (const char *)a1[2]);
    v8 = *(_WORD **)(v7 + 24);
    v9 = *(_QWORD *)(v7 + 16) - (_QWORD)v8;
    if ( v9 < a3 )
    {
      v25 = sub_16E7EE0(v7, a2, a3);
      v8 = *(_WORD **)(v25 + 24);
      v7 = v25;
      v9 = *(_QWORD *)(v25 + 16) - (_QWORD)v8;
    }
    else if ( a3 )
    {
      memcpy(v8, a2, a3);
      v26 = *(_QWORD *)(v7 + 16);
      v8 = (_WORD *)(a3 + *(_QWORD *)(v7 + 24));
      *(_QWORD *)(v7 + 24) = v8;
      v9 = v26 - (_QWORD)v8;
    }
    if ( v9 <= 1 )
    {
      sub_16E7EE0(v7, ": ", 2);
    }
    else
    {
      *v8 = 8250;
      *(_QWORD *)(v7 + 24) += 2LL;
    }
    v31 = (unsigned int *)v33;
    v32 = 0x800000000LL;
    v10 = sub_15B16B0(a4, &v31);
    v11 = v31;
    v28 = v10;
    v12 = &v31[(unsigned int)v32];
    if ( v31 != v12 )
    {
      v13 = v31;
      v14 = 1;
      while ( 1 )
      {
        v16 = sub_15B08C0(*v13);
        v17 = *a1;
        v18 = (const char *)v16;
        v20 = v19;
        if ( v14 )
          break;
        v21 = *(_QWORD *)(v17 + 24);
        if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v21) <= 2 )
        {
          v27 = v20;
          v30 = (const char *)v16;
          v23 = sub_16E7EE0(*a1, " | ", 3);
          v18 = v30;
          v20 = v27;
          v15 = *(void **)(v23 + 24);
          v17 = v23;
LABEL_12:
          if ( *(_QWORD *)(v17 + 16) - (_QWORD)v15 >= v20 )
            goto LABEL_13;
LABEL_19:
          ++v13;
          sub_16E7EE0(v17, v18, v20);
          if ( v12 == v13 )
            goto LABEL_20;
        }
        else
        {
          *(_BYTE *)(v21 + 2) = 32;
          *(_WORD *)v21 = 31776;
          v15 = (void *)(*(_QWORD *)(v17 + 24) + 3LL);
          v22 = *(_QWORD *)(v17 + 16);
          *(_QWORD *)(v17 + 24) = v15;
          if ( v22 - (__int64)v15 < v20 )
            goto LABEL_19;
LABEL_13:
          if ( v20 )
          {
            v29 = v20;
            memcpy(v15, v18, v20);
            *(_QWORD *)(v17 + 24) += v29;
          }
          if ( v12 == ++v13 )
          {
LABEL_20:
            if ( !v28 && (_DWORD)v32 )
            {
LABEL_22:
              v11 = v31;
              goto LABEL_23;
            }
            v24 = sub_1263B40(*a1, " | ");
LABEL_29:
            sub_16E7A90(v24, v28);
            goto LABEL_22;
          }
        }
      }
      v15 = *(void **)(v17 + 24);
      v14 = 0;
      goto LABEL_12;
    }
    if ( v10 || !(_DWORD)v32 )
    {
      v24 = *a1;
      goto LABEL_29;
    }
LABEL_23:
    if ( v11 != (unsigned int *)v33 )
      _libc_free((unsigned __int64)v11);
  }
}

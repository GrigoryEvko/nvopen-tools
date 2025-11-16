// Function: sub_2215540
// Address: 0x2215540
//
void __fastcall sub_2215540(volatile signed __int32 **a1, size_t a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r8
  size_t v6; // rcx
  __int64 v9; // r12
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // r15
  size_t v12; // r12
  __int64 v13; // rax
  size_t v14; // rcx
  char *v15; // r14
  volatile signed __int32 *v16; // r9
  volatile signed __int32 *v17; // rsi
  volatile signed __int32 *v18; // rax
  volatile signed __int32 *v19; // rax
  volatile signed __int32 *v20; // rdi
  char *v21; // rcx
  char *v22; // rdi
  _BYTE *v23; // rsi
  _BYTE *v24; // rdi
  int v25; // edx
  size_t v26; // [rsp+0h] [rbp-58h]
  volatile signed __int32 *v27; // [rsp+0h] [rbp-58h]
  volatile signed __int32 *v28; // [rsp+0h] [rbp-58h]

  v4 = a4 - a3;
  v6 = a3 + a2;
  v9 = *((_QWORD *)*a1 - 3);
  v10 = *((_QWORD *)*a1 - 2);
  v11 = v4 + v9;
  v12 = v9 - v6;
  if ( v11 > v10 )
  {
LABEL_4:
    v26 = v6;
    v13 = sub_22153F0(v11, v10);
    v14 = v26;
    v15 = (char *)v13;
    v16 = (volatile signed __int32 *)(v13 + 24);
    if ( a2 )
    {
      v17 = *a1;
      v16 = (volatile signed __int32 *)(v13 + 24);
      if ( a2 == 1 )
      {
        *(_BYTE *)(v13 + 24) = *(_BYTE *)v17;
        if ( !v12 )
          goto LABEL_8;
        goto LABEL_12;
      }
      v18 = (volatile signed __int32 *)memcpy((void *)(v13 + 24), (const void *)v17, a2);
      v14 = v26;
      v16 = v18;
    }
    if ( !v12 )
    {
LABEL_8:
      v19 = *a1;
      v20 = *a1 - 6;
      if ( v20 != (volatile signed __int32 *)&unk_4FD67C0 )
      {
        if ( &_pthread_key_create )
        {
          v25 = _InterlockedExchangeAdd(v19 - 2, 0xFFFFFFFF);
        }
        else
        {
          v25 = *((_DWORD *)v19 - 2);
          *((_DWORD *)v19 - 2) = v25 - 1;
        }
        if ( v25 <= 0 )
        {
          v28 = v16;
          j_j___libc_free_0_1((unsigned __int64)v20);
          v16 = v28;
        }
      }
      *a1 = v16;
LABEL_10:
      if ( v15 == (char *)&unk_4FD67C0 )
        return;
LABEL_19:
      *((_DWORD *)v16 - 2) = 0;
      *((_QWORD *)v16 - 3) = v11;
      *((_BYTE *)v16 + v11) = 0;
      return;
    }
LABEL_12:
    v21 = (char *)*a1 + v14;
    v22 = &v15[a2 + 24 + a4];
    if ( v12 == 1 )
    {
      *v22 = *v21;
    }
    else
    {
      v27 = v16;
      memcpy(v22, v21, v12);
      v16 = v27;
    }
    goto LABEL_8;
  }
  if ( *((int *)*a1 - 2) > 0 )
  {
    v10 = *((_QWORD *)*a1 - 2);
    goto LABEL_4;
  }
  v16 = *a1;
  if ( v12 && a4 != a3 )
  {
    v23 = (char *)v16 + v6;
    v24 = (char *)v16 + a4 + a2;
    if ( v12 == 1 )
    {
      *v24 = *v23;
      v16 = *a1;
      v15 = (char *)(*a1 - 6);
      goto LABEL_10;
    }
    memmove(v24, v23, v12);
    v16 = *a1;
  }
  if ( v16 - 6 != (volatile signed __int32 *)&unk_4FD67C0 )
    goto LABEL_19;
}

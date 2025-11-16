// Function: sub_22161B0
// Address: 0x22161b0
//
void __fastcall sub_22161B0(const wchar_t **a1, size_t a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  size_t v6; // rcx
  __int64 v9; // r12
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // r14
  size_t v12; // r12
  __int64 v13; // rax
  size_t v14; // rcx
  const wchar_t *v15; // r8
  const wchar_t *v16; // rsi
  const wchar_t *v17; // r9
  wchar_t *v18; // rdi
  const wchar_t *v19; // rsi
  wchar_t *v20; // rdi
  int v21; // eax
  size_t v22; // [rsp+8h] [rbp-60h]
  const wchar_t *v23; // [rsp+8h] [rbp-60h]
  __int64 v24; // [rsp+8h] [rbp-60h]
  const wchar_t *v25; // [rsp+8h] [rbp-60h]
  const wchar_t *v26; // [rsp+10h] [rbp-58h]
  const wchar_t *v27; // [rsp+10h] [rbp-58h]
  const wchar_t *v28; // [rsp+10h] [rbp-58h]
  size_t v29; // [rsp+18h] [rbp-50h]

  v4 = a4 - a3;
  v6 = a3 + a2;
  v9 = *((_QWORD *)*a1 - 3);
  v10 = *((_QWORD *)*a1 - 2);
  v11 = v9 + v4;
  v12 = v9 - v6;
  if ( v11 > v10 )
  {
LABEL_4:
    v22 = v6;
    v13 = sub_2216040(v11, v10);
    v14 = v22;
    v15 = (const wchar_t *)v13;
    if ( a2 )
    {
      v16 = *a1;
      v17 = (const wchar_t *)(v13 + 24);
      if ( a2 == 1 )
      {
        *(_DWORD *)(v13 + 24) = *v16;
      }
      else
      {
        v29 = v22;
        v24 = v13 + 24;
        v27 = (const wchar_t *)v13;
        wmemcpy((wchar_t *)(v13 + 24), v16, a2);
        v16 = *a1;
        v14 = v29;
        v15 = v27;
        v17 = (const wchar_t *)v24;
      }
    }
    else
    {
      v16 = *a1;
      v17 = (const wchar_t *)(v13 + 24);
    }
    if ( v12 )
    {
      v18 = (wchar_t *)&v15[a4 + 6 + a2];
      if ( v12 == 1 )
      {
        *v18 = v16[v14];
      }
      else
      {
        v26 = v17;
        v23 = v15;
        wmemcpy(v18, &v16[v14], v12);
        v16 = *a1;
        v17 = v26;
        v15 = v23;
      }
    }
    if ( v16 - 6 != (const wchar_t *)&unk_4FD67E0 )
    {
      if ( &_pthread_key_create )
      {
        v21 = _InterlockedExchangeAdd((volatile signed __int32 *)v16 - 2, 0xFFFFFFFF);
      }
      else
      {
        v21 = *(v16 - 2);
        *((_DWORD *)v16 - 2) = v21 - 1;
      }
      if ( v21 <= 0 )
      {
        v28 = v17;
        v25 = v15;
        j_j___libc_free_0_2((unsigned __int64)(v16 - 6));
        v17 = v28;
        v15 = v25;
      }
    }
    *a1 = v17;
LABEL_11:
    if ( v15 == (const wchar_t *)&unk_4FD67E0 )
      return;
LABEL_20:
    *((_DWORD *)v17 - 2) = 0;
    *((_QWORD *)v17 - 3) = v11;
    v17[v11] = 0;
    return;
  }
  if ( *(*a1 - 2) > 0 )
  {
    v10 = *((_QWORD *)*a1 - 2);
    goto LABEL_4;
  }
  v17 = *a1;
  if ( v12 && a4 != a3 )
  {
    v19 = &v17[v6];
    v20 = (wchar_t *)&v17[a4 + a2];
    if ( v12 == 1 )
    {
      v15 = v17 - 6;
      *v20 = *v19;
      goto LABEL_11;
    }
    wmemmove(v20, v19, v12);
    v17 = *a1;
  }
  if ( v17 - 6 != (const wchar_t *)&unk_4FD67E0 )
    goto LABEL_20;
}

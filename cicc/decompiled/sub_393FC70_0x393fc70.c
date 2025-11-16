// Function: sub_393FC70
// Address: 0x393fc70
//
void __fastcall sub_393FC70(__int64 a1)
{
  __int64 v1; // r12
  char *v2; // rax
  char *v3; // rbx
  void *v4; // rcx
  int v5; // ecx
  _QWORD *v6; // rsi
  __int64 *v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // rbx
  __int64 *v10; // r12
  __int64 v11; // rax
  __int64 *v12; // rdx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // [rsp+8h] [rbp-A8h] BYREF
  __int64 v21; // [rsp+10h] [rbp-A0h] BYREF
  int v22; // [rsp+18h] [rbp-98h] BYREF
  unsigned __int64 v23; // [rsp+20h] [rbp-90h]
  int *v24; // [rsp+28h] [rbp-88h]
  int *v25; // [rsp+30h] [rbp-80h]
  __int64 v26; // [rsp+38h] [rbp-78h]
  unsigned __int64 v27; // [rsp+40h] [rbp-70h]
  char *v28; // [rsp+48h] [rbp-68h]
  char *v29; // [rsp+50h] [rbp-60h]
  unsigned __int64 v30; // [rsp+58h] [rbp-58h]
  __int64 v31; // [rsp+60h] [rbp-50h]
  __int64 v32; // [rsp+68h] [rbp-48h]
  __int64 v33; // [rsp+70h] [rbp-40h]
  __int64 v34; // [rsp+78h] [rbp-38h]
  __int64 v35; // [rsp+80h] [rbp-30h]
  __int64 v36; // [rsp+88h] [rbp-28h]

  v1 = 4LL * unk_49D9478;
  if ( (unsigned __int64)(4LL * unk_49D9478) > 0x7FFFFFFFFFFFFFFCLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v1 )
  {
    v2 = (char *)sub_22077B0(4LL * unk_49D9478);
    v3 = &v2[v1];
    v4 = memcpy(v2, &unk_4530740, 4LL * unk_49D9478);
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v27 = (unsigned __int64)v4;
  v5 = *(_DWORD *)(a1 + 16);
  v22 = 0;
  v23 = 0;
  v24 = &v22;
  v25 = &v22;
  v26 = 0;
  v28 = v3;
  v29 = v3;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  if ( v5 )
  {
    v6 = *(_QWORD **)(a1 + 8);
    if ( *v6 != -8 && *v6 )
    {
      v9 = *(__int64 **)(a1 + 8);
    }
    else
    {
      v7 = v6 + 1;
      do
      {
        do
        {
          v8 = *v7;
          v9 = v7++;
        }
        while ( v8 == -8 );
      }
      while ( !v8 );
    }
    v10 = &v6[v5];
    while ( v9 != v10 )
    {
      while ( 1 )
      {
        sub_393CD60((__int64)&v21, *v9 + 8);
        v11 = v9[1];
        v12 = v9 + 1;
        if ( v11 == -8 || !v11 )
          break;
        ++v9;
        if ( v12 == v10 )
          goto LABEL_17;
      }
      v13 = v9 + 2;
      do
      {
        do
        {
          v14 = *v13;
          v9 = v13++;
        }
        while ( !v14 );
      }
      while ( v14 == -8 );
    }
  }
LABEL_17:
  sub_393CA80((__int64 *)&v20, &v21);
  v15 = v20;
  v16 = *(_QWORD *)(a1 + 56);
  v20 = 0;
  *(_QWORD *)(a1 + 56) = v15;
  if ( v16 )
  {
    v17 = *(_QWORD *)(v16 + 8);
    if ( v17 )
      j_j___libc_free_0(v17);
    j_j___libc_free_0(v16);
    v18 = v20;
    if ( v20 )
    {
      v19 = *(_QWORD *)(v20 + 8);
      if ( v19 )
        j_j___libc_free_0(v19);
      j_j___libc_free_0(v18);
    }
  }
  if ( v30 )
    j_j___libc_free_0(v30);
  if ( v27 )
    j_j___libc_free_0(v27);
  sub_393DD20(v23);
}

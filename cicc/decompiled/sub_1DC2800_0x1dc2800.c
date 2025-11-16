// Function: sub_1DC2800
// Address: 0x1dc2800
//
void __fastcall sub_1DC2800(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  unsigned int v4; // r15d
  void *v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // r8d
  int v9; // r9d
  unsigned __int16 *v10; // r15
  unsigned int v11; // esi
  unsigned int *v12; // r15
  unsigned int *i; // r12
  unsigned int v14; // esi
  unsigned __int16 *v15; // r12
  unsigned __int16 *v16; // r13
  unsigned int v17; // esi
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  unsigned __int16 *v22; // r12
  unsigned int v23; // esi
  unsigned int *v24; // r12
  unsigned int *v25; // r13
  unsigned int v26; // esi
  __int64 v27; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int16 *v28; // [rsp+18h] [rbp-78h]
  __int64 v29; // [rsp+20h] [rbp-70h]
  _BYTE v30[32]; // [rsp+28h] [rbp-68h] BYREF
  unsigned __int64 v31; // [rsp+48h] [rbp-48h]
  unsigned int v32; // [rsp+50h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 56);
  if ( *(_BYTE *)(v2 + 104) )
  {
    if ( *(_DWORD *)(a1 + 16) )
    {
      v3 = *(_QWORD *)a1;
      v31 = 0;
      v29 = 0x800000000LL;
      v32 = 0;
      v4 = *(_DWORD *)(v3 + 16);
      v27 = v3;
      v28 = (unsigned __int16 *)v30;
      if ( v4 )
      {
        v5 = _libc_calloc(v4, 1u);
        if ( !v5 )
        {
          sub_16BD1C0("Allocation failed", 1u);
          v5 = 0;
        }
        v31 = (unsigned __int64)v5;
        v32 = v4;
      }
      v10 = (unsigned __int16 *)sub_1E6A620(*(_QWORD *)(a2 + 40));
      if ( v10 )
      {
        while ( 1 )
        {
          v11 = *v10;
          if ( !(_WORD)v11 )
            break;
          ++v10;
          sub_1DC1BF0(&v27, v11, v6, v7, v8, v9);
        }
      }
      v12 = *(unsigned int **)(v2 + 88);
      for ( i = *(unsigned int **)(v2 + 80); v12 != i; i += 3 )
      {
        v14 = *i;
        sub_1DC1CF0((__int64)&v27, v14);
      }
      v15 = v28;
      v16 = &v28[2 * (unsigned int)v29];
      if ( v16 != v28 )
      {
        do
        {
          v17 = *v15;
          v15 += 2;
          sub_1DC1BF0((__int64 *)a1, v17, v6, v7, v8, v9);
        }
        while ( v16 != v15 );
      }
      _libc_free(v31);
      if ( v28 != (unsigned __int16 *)v30 )
        _libc_free((unsigned __int64)v28);
    }
    else
    {
      v22 = (unsigned __int16 *)sub_1E6A620(*(_QWORD *)(a2 + 40));
      if ( v22 )
      {
        while ( 1 )
        {
          v23 = *v22;
          if ( !(_WORD)v23 )
            break;
          ++v22;
          sub_1DC1BF0((__int64 *)a1, v23, v18, v19, v20, v21);
        }
      }
      v24 = *(unsigned int **)(v2 + 80);
      v25 = *(unsigned int **)(v2 + 88);
      while ( v25 != v24 )
      {
        v26 = *v24;
        v24 += 3;
        sub_1DC1CF0(a1, v26);
      }
    }
  }
}

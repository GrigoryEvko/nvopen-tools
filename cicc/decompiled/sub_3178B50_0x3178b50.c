// Function: sub_3178B50
// Address: 0x3178b50
//
void __fastcall sub_3178B50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r13
  __int64 v5; // r12
  unsigned int v6; // eax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rbx
  _DWORD *v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  unsigned __int64 v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r12
  int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // rdi
  __int64 v30; // [rsp+8h] [rbp-118h]
  __int64 v31; // [rsp+10h] [rbp-110h]
  _DWORD *v32; // [rsp+28h] [rbp-F8h] BYREF
  int v33; // [rsp+30h] [rbp-F0h]
  char *v34; // [rsp+38h] [rbp-E8h]
  __int64 v35; // [rsp+40h] [rbp-E0h]
  char v36; // [rsp+48h] [rbp-D8h] BYREF
  int v37; // [rsp+90h] [rbp-90h]
  _BYTE *v38; // [rsp+98h] [rbp-88h] BYREF
  __int64 v39; // [rsp+A0h] [rbp-80h]
  _BYTE v40[120]; // [rsp+A8h] [rbp-78h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  v7 = sub_C7D670(96LL * v6, 8);
  *(_QWORD *)(a1 + 8) = v7;
  v12 = v7;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v38 = v40;
    v30 = 96 * v4;
    v13 = v5 + 96 * v4;
    v39 = 0x400000000LL;
    v14 = *(unsigned int *)(a1 + 24);
    v37 = -1;
    v15 = v12 + 96 * v14;
    if ( v12 != v15 )
    {
      do
      {
        while ( 1 )
        {
          if ( v12 )
          {
            v16 = v37;
            *(_DWORD *)(v12 + 16) = 0;
            *(_DWORD *)(v12 + 20) = 4;
            *(_DWORD *)v12 = v16;
            *(_QWORD *)(v12 + 8) = v12 + 24;
            if ( (_DWORD)v39 )
              break;
          }
          v12 += 96;
          if ( v15 == v12 )
            goto LABEL_10;
        }
        v17 = v12 + 8;
        v31 = v15;
        v12 += 96;
        sub_3174B60(v17, (__int64)&v38, (unsigned int)v39, v9, v10, v11);
        v15 = v31;
      }
      while ( v31 != v12 );
LABEL_10:
      if ( v38 != v40 )
        _libc_free((unsigned __int64)v38);
    }
    v38 = v40;
    v34 = &v36;
    v33 = -1;
    v35 = 0x400000000LL;
    v37 = -2;
    v39 = 0x400000000LL;
    if ( v13 != v5 )
    {
      v18 = v5;
      do
      {
        if ( *(_DWORD *)v18 < 0xFFFFFFFE || *(_DWORD *)(v18 + 16) )
        {
          sub_3178240(a1, (unsigned int *)v18, &v32);
          v19 = v32;
          *v32 = *(_DWORD *)v18;
          sub_3174A00((__int64)(v19 + 2), (char **)(v18 + 8), v20, v21, v22, v23);
          v32[22] = *(_DWORD *)(v18 + 88);
          ++*(_DWORD *)(a1 + 16);
        }
        v24 = *(_QWORD *)(v18 + 8);
        if ( v24 != v18 + 24 )
          _libc_free(v24);
        v18 += 96;
      }
      while ( v13 != v18 );
    }
    sub_C7D6A0(v5, v30, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    v39 = 0x400000000LL;
    v25 = *(unsigned int *)(a1 + 24);
    v37 = -1;
    v38 = v40;
    v26 = v12 + 96 * v25;
    if ( v12 != v26 )
    {
      do
      {
        while ( 1 )
        {
          if ( v12 )
          {
            v27 = v37;
            v28 = (unsigned int)v39;
            *(_DWORD *)(v12 + 16) = 0;
            *(_DWORD *)(v12 + 20) = 4;
            *(_DWORD *)v12 = v27;
            *(_QWORD *)(v12 + 8) = v12 + 24;
            if ( (_DWORD)v28 )
              break;
          }
          v12 += 96;
          if ( v26 == v12 )
            goto LABEL_27;
        }
        v29 = v12 + 8;
        v12 += 96;
        sub_3174B60(v29, (__int64)&v38, v8, v28, v10, v11);
      }
      while ( v26 != v12 );
LABEL_27:
      if ( v38 != v40 )
        _libc_free((unsigned __int64)v38);
    }
  }
}

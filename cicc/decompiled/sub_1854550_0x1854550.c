// Function: sub_1854550
// Address: 0x1854550
//
void __fastcall sub_1854550(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // r8d
  _QWORD *v5; // rax
  _QWORD *v8; // r12
  _QWORD *v9; // rbx
  _BYTE **v10; // r8
  __int64 v11; // rdi
  int v12; // eax
  int v13; // edx
  __int64 *i; // rax
  unsigned __int64 v15; // rcx
  int v16; // r10d
  _QWORD *v17; // rdi
  _BYTE **v19; // [rsp+18h] [rbp-C68h]
  __int64 *v20; // [rsp+18h] [rbp-C68h]
  __int64 v21; // [rsp+20h] [rbp-C60h] BYREF
  __int64 v22; // [rsp+28h] [rbp-C58h]
  __int64 v23; // [rsp+30h] [rbp-C50h]
  int v24; // [rsp+38h] [rbp-C48h]
  _BYTE *v25; // [rsp+40h] [rbp-C40h] BYREF
  __int64 v26; // [rsp+48h] [rbp-C38h]
  _BYTE v27[3120]; // [rsp+50h] [rbp-C30h] BYREF

  v4 = *(_DWORD *)(a1 + 16);
  v25 = v27;
  v26 = 0x8000000000LL;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  if ( v4 )
  {
    v5 = *(_QWORD **)(a1 + 8);
    v8 = &v5[2 * *(unsigned int *)(a1 + 24)];
    if ( v5 != v8 )
    {
      while ( 1 )
      {
        v9 = v5;
        if ( *v5 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v5 += 2;
        if ( v8 == v5 )
          goto LABEL_2;
      }
      if ( v5 != v8 )
      {
        v10 = &v25;
        do
        {
          v11 = v9[1];
          if ( !*(_BYTE *)(a2 + 176) || (*(_BYTE *)(v11 + 12) & 0x20) != 0 )
          {
            v12 = *(_DWORD *)(v11 + 8);
            if ( !v12 )
            {
              v11 = *(_QWORD *)(v11 + 64);
              v12 = *(_DWORD *)(v11 + 8);
            }
            if ( v12 == 1 )
            {
              v19 = v10;
              sub_1853180((_QWORD *)v11, a2, dword_4FAB200, a1, (__int64)v10, a3, a4, (__int64)&v21);
              v10 = v19;
            }
          }
          v9 += 2;
          if ( v9 == v8 )
            break;
          while ( *v9 > 0xFFFFFFFFFFFFFFFDLL )
          {
            v9 += 2;
            if ( v8 == v9 )
              goto LABEL_20;
          }
        }
        while ( v9 != v8 );
LABEL_20:
        v13 = v26;
        for ( i = &v21; (_DWORD)v26; i = v20 )
        {
          v20 = i;
          v15 = (unsigned __int64)&v25[24 * v13 - 24];
          v16 = *(_DWORD *)(v15 + 8);
          v17 = *(_QWORD **)(v15 + 16);
          LODWORD(v26) = v13 - 1;
          sub_1853180(v17, a2, v16, a1, (__int64)&v25, a3, a4, (__int64)i);
          v13 = v26;
        }
      }
    }
  }
LABEL_2:
  j___libc_free_0(v22);
  if ( v25 != v27 )
    _libc_free((unsigned __int64)v25);
}

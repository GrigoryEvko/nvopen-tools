// Function: sub_2F8B420
// Address: 0x2f8b420
//
__int64 __fastcall sub_2F8B420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r8
  __int64 v12; // r9
  char *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // r12
  __int64 v19; // r15
  __int64 i; // r13
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // r8
  __int64 v24; // r9
  char *v25; // rdi
  __int64 v26; // rcx
  __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  char *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // [rsp+8h] [rbp-B8h]
  __int64 v36; // [rsp+10h] [rbp-B0h]
  __int64 v37; // [rsp+18h] [rbp-A8h]
  __int64 v38; // [rsp+20h] [rbp-A0h]
  __int64 v39; // [rsp+28h] [rbp-98h]
  __int64 v40; // [rsp+30h] [rbp-90h]
  __int64 v41; // [rsp+30h] [rbp-90h]
  __int64 v42; // [rsp+30h] [rbp-90h]
  int v43; // [rsp+38h] [rbp-88h]
  int v44; // [rsp+38h] [rbp-88h]
  int v45; // [rsp+38h] [rbp-88h]
  unsigned __int8 v46; // [rsp+3Ch] [rbp-84h]
  unsigned __int8 v47; // [rsp+3Ch] [rbp-84h]
  char v48; // [rsp+3Ch] [rbp-84h]
  char *v49; // [rsp+40h] [rbp-80h] BYREF
  __int64 v50; // [rsp+48h] [rbp-78h]
  _BYTE v51[48]; // [rsp+50h] [rbp-70h] BYREF
  int v52; // [rsp+80h] [rbp-40h]

  v36 = a1;
  if ( a1 == a2 )
    return a3;
  if ( a2 == a3 )
    return a1;
  v35 = a1 + a3 - a2;
  v37 = 0x2E8BA2E8BA2E8BA3LL * ((a3 - a1) >> 3);
  v38 = 0x2E8BA2E8BA2E8BA3LL * ((a2 - a1) >> 3);
  if ( v38 != v37 - v38 )
  {
    while ( 1 )
    {
      v39 = v37 - v38;
      if ( v38 >= v37 - v38 )
      {
        v16 = v36 + 88 * v37;
        v17 = 88 * v39;
        v36 = v16 - 88 * v39;
        if ( v38 > 0 )
        {
          v18 = v16 - 88 * v39 - 72;
          v19 = v16 - 72;
          for ( i = 0; i != v38; ++i )
          {
            v26 = *(_QWORD *)(v18 - 16);
            v49 = v51;
            v50 = 0x600000000LL;
            v41 = v26;
            v44 = *(_DWORD *)(v18 - 8);
            v47 = *(_BYTE *)(v18 - 4);
            if ( *(_DWORD *)(v18 + 8) )
              sub_2F8ABB0((__int64)&v49, (char **)v18, v17, *(unsigned __int8 *)(v18 - 4), a5, a6);
            v52 = *(_DWORD *)(v18 + 64);
            *(_QWORD *)(v18 - 16) = *(_QWORD *)(v19 - 16);
            *(_DWORD *)(v18 - 8) = *(_DWORD *)(v19 - 8);
            v21 = *(unsigned __int8 *)(v19 - 4);
            *(_BYTE *)(v18 - 4) = v21;
            sub_2F8ABB0(v18, (char **)v19, v17, v21, a5, a6);
            *(_DWORD *)(v18 + 64) = *(_DWORD *)(v19 + 64);
            *(_QWORD *)(v19 - 16) = v41;
            *(_DWORD *)(v19 - 8) = v44;
            *(_BYTE *)(v19 - 4) = v47;
            sub_2F8ABB0(v19, &v49, v22, v47, v23, v24);
            v25 = v49;
            *(_DWORD *)(v19 + 64) = v52;
            if ( v25 != v51 )
              _libc_free((unsigned __int64)v25);
            v18 -= 88;
            v19 -= 88;
          }
          v36 += -88 * v38;
        }
        v38 = v37 % v39;
        if ( !(v37 % v39) )
          return v35;
      }
      else
      {
        if ( v37 - v38 > 0 )
        {
          v6 = 0;
          v7 = v36 + 16;
          v8 = v36 + 88 * v38 + 16;
          do
          {
            v14 = *(_QWORD *)(v7 - 16);
            v15 = *(unsigned int *)(v7 + 8);
            v49 = v51;
            v50 = 0x600000000LL;
            v40 = v14;
            v43 = *(_DWORD *)(v7 - 8);
            v46 = *(_BYTE *)(v7 - 4);
            if ( (_DWORD)v15 )
              sub_2F8ABB0((__int64)&v49, (char **)v7, v15, *(unsigned __int8 *)(v7 - 4), a5, a6);
            v52 = *(_DWORD *)(v7 + 64);
            *(_QWORD *)(v7 - 16) = *(_QWORD *)(v8 - 16);
            *(_DWORD *)(v7 - 8) = *(_DWORD *)(v8 - 8);
            v9 = *(unsigned __int8 *)(v8 - 4);
            *(_BYTE *)(v7 - 4) = v9;
            sub_2F8ABB0(v7, (char **)v8, v15, v9, a5, a6);
            *(_DWORD *)(v7 + 64) = *(_DWORD *)(v8 + 64);
            *(_QWORD *)(v8 - 16) = v40;
            *(_DWORD *)(v8 - 8) = v43;
            *(_BYTE *)(v8 - 4) = v46;
            sub_2F8ABB0(v8, &v49, v10, v46, v11, v12);
            v13 = v49;
            *(_DWORD *)(v8 + 64) = v52;
            if ( v13 != v51 )
              _libc_free((unsigned __int64)v13);
            ++v6;
            v7 += 88;
            v8 += 88;
          }
          while ( v39 != v6 );
          v36 += 88 * v39;
        }
        if ( !(v37 % v38) )
          return v35;
        v39 = v38;
        v38 -= v37 % v38;
      }
      v37 = v39;
    }
  }
  v35 = a1;
  v28 = a2 + 16;
  do
  {
    v49 = v51;
    v50 = 0x600000000LL;
    v33 = *(unsigned int *)(v35 + 24);
    v34 = v35 + 16;
    v42 = *(_QWORD *)v35;
    v45 = *(_DWORD *)(v35 + 8);
    v48 = *(_BYTE *)(v35 + 12);
    if ( (_DWORD)v33 )
    {
      sub_2F8ABB0((__int64)&v49, (char **)(v35 + 16), v35, v33, v34, a6);
      v34 = v35 + 16;
    }
    v52 = *(_DWORD *)(v35 + 80);
    *(_QWORD *)v35 = *(_QWORD *)(v28 - 16);
    *(_DWORD *)(v35 + 8) = *(_DWORD *)(v28 - 8);
    *(_BYTE *)(v35 + 12) = *(_BYTE *)(v28 - 4);
    sub_2F8ABB0(v34, (char **)v28, v35, v33, v34, a6);
    *(_DWORD *)(v35 + 80) = *(_DWORD *)(v28 + 64);
    *(_QWORD *)(v28 - 16) = v42;
    *(_DWORD *)(v28 - 8) = v45;
    *(_BYTE *)(v28 - 4) = v48;
    sub_2F8ABB0(v28, &v49, v35, v29, v30, v31);
    v32 = v49;
    *(_DWORD *)(v28 + 64) = v52;
    if ( v32 != v51 )
      _libc_free((unsigned __int64)v32);
    v35 += 88;
    v28 += 88;
  }
  while ( a2 != v35 );
  return v35;
}

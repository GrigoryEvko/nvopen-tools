// Function: sub_2EB9660
// Address: 0x2eb9660
//
__int64 __fastcall sub_2EB9660(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int8 (__fastcall *a4)(__int64, _QWORD),
        __int64 a5,
        __int64 *a6)
{
  __int64 v6; // r8
  __int64 v7; // r9
  unsigned int i; // eax
  __int64 v9; // rdx
  __int64 v10; // rbx
  int v11; // r14d
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rax
  char *v20; // r14
  char *v21; // r13
  char *v22; // r12
  unsigned __int64 v23; // r14
  __int64 v24; // r15
  __int64 v25; // rax
  unsigned __int64 v26; // r10
  unsigned __int64 v27; // rdx
  _QWORD *v28; // rax
  __int64 v30; // r13
  __int64 *v31; // r14
  __int64 *v32; // r15
  unsigned __int64 v33; // rax
  __int64 *v34; // r13
  __int64 v35; // rbx
  __int64 v36; // r15
  unsigned int v37; // r14d
  __int64 v38; // [rsp+0h] [rbp-530h]
  __int64 *v39; // [rsp+8h] [rbp-528h]
  unsigned __int64 v40; // [rsp+18h] [rbp-518h]
  __int64 *v42; // [rsp+28h] [rbp-508h]
  unsigned __int64 v43; // [rsp+48h] [rbp-4E8h]
  __int64 *v44; // [rsp+48h] [rbp-4E8h]
  unsigned int v45; // [rsp+54h] [rbp-4DCh]
  __int64 *v47[4]; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 *v48[4]; // [rsp+80h] [rbp-4B0h] BYREF
  void *src; // [rsp+A0h] [rbp-490h] BYREF
  unsigned int v50; // [rsp+A8h] [rbp-488h]
  char v51; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD *v52; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v53; // [rsp+F8h] [rbp-438h]
  _QWORD v54[134]; // [rsp+100h] [rbp-430h] BYREF

  v52 = v54;
  v54[1] = (unsigned int)a5;
  v45 = a3;
  v54[0] = a2;
  v53 = 0x4000000001LL;
  *(_DWORD *)(sub_2EB5B40(a1, a2, a3, (__int64)a4, a5, (__int64)a6) + 4) = a5;
  for ( i = v53; (_DWORD)v53; i = v53 )
  {
    while ( 1 )
    {
      v9 = (__int64)&v52[2 * i - 2];
      v10 = *(_QWORD *)v9;
      v11 = *(_DWORD *)(v9 + 8);
      LODWORD(v53) = i - 1;
      v12 = sub_2EB5B40(a1, v10, v9, (__int64)v52, v6, v7);
      v13 = *(unsigned int *)(v12 + 32);
      v14 = *(unsigned int *)(v12 + 36);
      if ( v13 + 1 > v14 )
      {
        sub_C8D5F0(v12 + 24, (const void *)(v12 + 40), v13 + 1, 4u, v6, v7);
        v13 = *(unsigned int *)(v12 + 32);
      }
      v15 = *(_QWORD *)(v12 + 24);
      *(_DWORD *)(v15 + 4 * v13) = v11;
      v16 = *(_DWORD *)v12;
      ++*(_DWORD *)(v12 + 32);
      if ( !v16 )
      {
        ++v45;
        *(_DWORD *)(v12 + 4) = v11;
        *(_DWORD *)(v12 + 12) = v45;
        *(_DWORD *)(v12 + 8) = v45;
        *(_DWORD *)v12 = v45;
        sub_2E6D5A0(a1, v10, v15, v14, v6, v7);
        sub_2EB5530(&src, v10, *(_QWORD *)(a1 + 4128), v17, v18);
        v19 = v50;
        if ( a6 && v50 > 1uLL )
        {
          v30 = 8LL * v50;
          v31 = (__int64 *)src;
          v32 = (__int64 *)((char *)src + v30);
          _BitScanReverse64(&v33, v30 >> 3);
          v39 = (__int64 *)((char *)src + v30);
          sub_2EB9180((__int64 *)src, (char *)src + v30, 2LL * (int)(63 - (v33 ^ 0x3F)), (__int64)a6);
          if ( (unsigned __int64)v30 <= 0x80 )
          {
            sub_2EB7B30(v31, v39, (__int64)a6);
          }
          else
          {
            v42 = v31 + 16;
            sub_2EB7B30(v31, v31 + 16, (__int64)a6);
            if ( v31 + 16 != v32 )
            {
              v38 = v10;
              do
              {
                v34 = v42;
                v35 = *v42;
                while ( 1 )
                {
                  v36 = *(v34 - 1);
                  v44 = v34--;
                  sub_2E6E850(v47, a6, v35);
                  v37 = *((_DWORD *)v47[2] + 2);
                  sub_2E6E850(v48, a6, v36);
                  if ( v37 >= *((_DWORD *)v48[2] + 2) )
                    break;
                  v34[1] = *v34;
                }
                ++v42;
                *v44 = v35;
              }
              while ( v42 != v39 );
              v10 = v38;
            }
          }
          v19 = v50;
        }
        v20 = (char *)src;
        v21 = (char *)src + 8 * v19;
        if ( v21 != src )
        {
          v22 = (char *)src;
          v23 = v40;
          do
          {
            v24 = *(_QWORD *)v22;
            if ( a4(v10, *(_QWORD *)v22) )
            {
              v25 = (unsigned int)v53;
              v26 = v23 & 0xFFFFFFFF00000000LL | v45;
              v27 = (unsigned int)v53 + 1LL;
              v23 = v26;
              if ( v27 > HIDWORD(v53) )
              {
                v43 = v26;
                sub_C8D5F0((__int64)&v52, v54, v27, 0x10u, v6, v7);
                v25 = (unsigned int)v53;
                v26 = v43;
              }
              v28 = &v52[2 * v25];
              *v28 = v24;
              v28[1] = v26;
              LODWORD(v53) = v53 + 1;
            }
            v22 += 8;
          }
          while ( v21 != v22 );
          v40 = v23;
          v20 = (char *)src;
        }
        if ( v20 != &v51 )
          break;
      }
      i = v53;
      if ( !(_DWORD)v53 )
        goto LABEL_19;
    }
    _libc_free((unsigned __int64)v20);
  }
LABEL_19:
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return v45;
}

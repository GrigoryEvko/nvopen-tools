// Function: sub_2EB8890
// Address: 0x2eb8890
//
__int64 __fastcall sub_2EB8890(
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
  __int64 v19; // r9
  __int64 v20; // rax
  char *v21; // r14
  char *v22; // r13
  char *v23; // r12
  unsigned __int64 v24; // r14
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned __int64 v27; // r10
  unsigned __int64 v28; // rdx
  _QWORD *v29; // rax
  __int64 v31; // r13
  __int64 *v32; // r14
  __int64 *v33; // r15
  unsigned __int64 v34; // rax
  __int64 *v35; // r13
  __int64 v36; // rbx
  __int64 v37; // r15
  unsigned int v38; // r14d
  __int64 v39; // [rsp+0h] [rbp-530h]
  __int64 *v40; // [rsp+8h] [rbp-528h]
  unsigned __int64 v41; // [rsp+18h] [rbp-518h]
  __int64 *v43; // [rsp+28h] [rbp-508h]
  unsigned __int64 v44; // [rsp+48h] [rbp-4E8h]
  __int64 *v45; // [rsp+48h] [rbp-4E8h]
  unsigned int v46; // [rsp+54h] [rbp-4DCh]
  __int64 *v48[4]; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 *v49[4]; // [rsp+80h] [rbp-4B0h] BYREF
  void *src; // [rsp+A0h] [rbp-490h] BYREF
  unsigned int v51; // [rsp+A8h] [rbp-488h]
  char v52; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD *v53; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v54; // [rsp+F8h] [rbp-438h]
  _QWORD v55[134]; // [rsp+100h] [rbp-430h] BYREF

  v53 = v55;
  v55[1] = (unsigned int)a5;
  v46 = a3;
  v55[0] = a2;
  v54 = 0x4000000001LL;
  *(_DWORD *)(sub_2EB5B40(a1, a2, a3, (__int64)a4, a5, (__int64)a6) + 4) = a5;
  for ( i = v54; (_DWORD)v54; i = v54 )
  {
    while ( 1 )
    {
      v9 = (__int64)&v53[2 * i - 2];
      v10 = *(_QWORD *)v9;
      v11 = *(_DWORD *)(v9 + 8);
      LODWORD(v54) = i - 1;
      v12 = sub_2EB5B40(a1, v10, v9, (__int64)v53, v6, v7);
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
        ++v46;
        *(_DWORD *)(v12 + 4) = v11;
        *(_DWORD *)(v12 + 12) = v46;
        *(_DWORD *)(v12 + 8) = v46;
        *(_DWORD *)v12 = v46;
        sub_2E6D5A0(a1, v10, v15, v14, v6, v7);
        sub_2EB52F0(&src, v10, *(_QWORD *)(a1 + 4128), v17, v18, v19);
        v20 = v51;
        if ( a6 && v51 > 1uLL )
        {
          v31 = 8LL * v51;
          v32 = (__int64 *)src;
          v33 = (__int64 *)((char *)src + v31);
          _BitScanReverse64(&v34, v31 >> 3);
          v40 = (__int64 *)((char *)src + v31);
          sub_2EB83B0((__int64 *)src, (char *)src + v31, 2LL * (int)(63 - (v34 ^ 0x3F)), (__int64)a6);
          if ( (unsigned __int64)v31 <= 0x80 )
          {
            sub_2EB7940(v32, v40, (__int64)a6);
          }
          else
          {
            v43 = v32 + 16;
            sub_2EB7940(v32, v32 + 16, (__int64)a6);
            if ( v32 + 16 != v33 )
            {
              v39 = v10;
              do
              {
                v35 = v43;
                v36 = *v43;
                while ( 1 )
                {
                  v37 = *(v35 - 1);
                  v45 = v35--;
                  sub_2E6E850(v48, a6, v36);
                  v38 = *((_DWORD *)v48[2] + 2);
                  sub_2E6E850(v49, a6, v37);
                  if ( v38 >= *((_DWORD *)v49[2] + 2) )
                    break;
                  v35[1] = *v35;
                }
                ++v43;
                *v45 = v36;
              }
              while ( v43 != v40 );
              v10 = v39;
            }
          }
          v20 = v51;
        }
        v21 = (char *)src;
        v22 = (char *)src + 8 * v20;
        if ( v22 != src )
        {
          v23 = (char *)src;
          v24 = v41;
          do
          {
            v25 = *(_QWORD *)v23;
            if ( a4(v10, *(_QWORD *)v23) )
            {
              v26 = (unsigned int)v54;
              v27 = v24 & 0xFFFFFFFF00000000LL | v46;
              v28 = (unsigned int)v54 + 1LL;
              v24 = v27;
              if ( v28 > HIDWORD(v54) )
              {
                v44 = v27;
                sub_C8D5F0((__int64)&v53, v55, v28, 0x10u, v6, v7);
                v26 = (unsigned int)v54;
                v27 = v44;
              }
              v29 = &v53[2 * v26];
              *v29 = v25;
              v29[1] = v27;
              LODWORD(v54) = v54 + 1;
            }
            v23 += 8;
          }
          while ( v22 != v23 );
          v41 = v24;
          v21 = (char *)src;
        }
        if ( v21 != &v52 )
          break;
      }
      i = v54;
      if ( !(_DWORD)v54 )
        goto LABEL_19;
    }
    _libc_free((unsigned __int64)v21);
  }
LABEL_19:
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  return v46;
}

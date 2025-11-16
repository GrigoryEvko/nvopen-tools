// Function: sub_2E74880
// Address: 0x2e74880
//
void __fastcall sub_2E74880(
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
  char *v20; // r13
  char *v21; // r12
  unsigned __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned __int64 v25; // r10
  unsigned __int64 v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // r13
  __int64 *v29; // r14
  __int64 *v30; // r15
  unsigned __int64 v31; // rax
  __int64 *v32; // r13
  __int64 v33; // rbx
  __int64 v34; // r15
  unsigned int v35; // r14d
  __int64 v36; // [rsp+0h] [rbp-530h]
  __int64 *v37; // [rsp+8h] [rbp-528h]
  unsigned __int64 v38; // [rsp+18h] [rbp-518h]
  __int64 *v40; // [rsp+28h] [rbp-508h]
  unsigned __int64 v41; // [rsp+48h] [rbp-4E8h]
  __int64 *v42; // [rsp+48h] [rbp-4E8h]
  unsigned int v43; // [rsp+54h] [rbp-4DCh]
  __int64 *v45[4]; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 *v46[4]; // [rsp+80h] [rbp-4B0h] BYREF
  void *src; // [rsp+A0h] [rbp-490h] BYREF
  unsigned int v48; // [rsp+A8h] [rbp-488h]
  char v49; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD *v50; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v51; // [rsp+F8h] [rbp-438h]
  _QWORD v52[134]; // [rsp+100h] [rbp-430h] BYREF

  v50 = v52;
  v52[1] = (unsigned int)a5;
  v43 = a3;
  v52[0] = a2;
  v51 = 0x4000000001LL;
  *(_DWORD *)(sub_2E6F1C0(a1, a2, a3, (__int64)a4, a5, (__int64)a6) + 4) = a5;
  for ( i = v51; (_DWORD)v51; i = v51 )
  {
    while ( 1 )
    {
      v9 = (__int64)&v50[2 * i - 2];
      v10 = *(_QWORD *)v9;
      v11 = *(_DWORD *)(v9 + 8);
      LODWORD(v51) = i - 1;
      v12 = sub_2E6F1C0(a1, v10, v9, (__int64)v50, v6, v7);
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
        ++v43;
        *(_DWORD *)(v12 + 4) = v11;
        *(_DWORD *)(v12 + 12) = v43;
        *(_DWORD *)(v12 + 8) = v43;
        *(_DWORD *)v12 = v43;
        sub_2E6D5A0(a1, v10, v15, v14, v6, v7);
        sub_2E6EC80(&src, v10, *(_QWORD *)(a1 + 4128), v17, v18);
        v19 = v48;
        if ( a6 && v48 > 1uLL )
        {
          v28 = 8LL * v48;
          v29 = (__int64 *)src;
          v30 = (__int64 *)((char *)src + v28);
          _BitScanReverse64(&v31, v28 >> 3);
          v37 = (__int64 *)((char *)src + v28);
          sub_2E743A0((__int64 *)src, (char *)src + v28, 2LL * (int)(63 - (v31 ^ 0x3F)), (__int64)a6);
          if ( (unsigned __int64)v28 <= 0x80 )
          {
            sub_2E72B10(v29, v37, (__int64)a6);
          }
          else
          {
            v40 = v29 + 16;
            sub_2E72B10(v29, v29 + 16, (__int64)a6);
            if ( v29 + 16 != v30 )
            {
              v36 = v10;
              do
              {
                v32 = v40;
                v33 = *v40;
                while ( 1 )
                {
                  v34 = *(v32 - 1);
                  v42 = v32--;
                  sub_2E6E850(v45, a6, v33);
                  v35 = *((_DWORD *)v45[2] + 2);
                  sub_2E6E850(v46, a6, v34);
                  if ( v35 >= *((_DWORD *)v46[2] + 2) )
                    break;
                  v32[1] = *v32;
                }
                ++v40;
                *v42 = v33;
              }
              while ( v40 != v37 );
              v10 = v36;
            }
          }
          v19 = v48;
        }
        v20 = (char *)src + 8 * v19;
        if ( src != v20 )
        {
          v21 = (char *)src;
          v22 = v38;
          do
          {
            v23 = *(_QWORD *)v21;
            if ( a4(v10, *(_QWORD *)v21) )
            {
              v24 = (unsigned int)v51;
              v25 = v22 & 0xFFFFFFFF00000000LL | v43;
              v26 = (unsigned int)v51 + 1LL;
              v22 = v25;
              if ( v26 > HIDWORD(v51) )
              {
                v41 = v25;
                sub_C8D5F0((__int64)&v50, v52, v26, 0x10u, v6, v7);
                v24 = (unsigned int)v51;
                v25 = v41;
              }
              v27 = &v50[2 * v24];
              *v27 = v23;
              v27[1] = v25;
              LODWORD(v51) = v51 + 1;
            }
            v21 += 8;
          }
          while ( v20 != v21 );
          v38 = v22;
          v20 = (char *)src;
        }
        if ( v20 != &v49 )
          break;
      }
      i = v51;
      if ( !(_DWORD)v51 )
        goto LABEL_19;
    }
    _libc_free((unsigned __int64)v20);
  }
LABEL_19:
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
}

// Function: sub_2EBC640
// Address: 0x2ebc640
//
__int64 __fastcall sub_2EBC640(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r14
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned int i; // eax
  __int64 v10; // rdx
  __int64 v11; // rbx
  int v12; // r13d
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  int v17; // eax
  __int64 *v18; // r15
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  char *v23; // rdi
  unsigned int v24; // r10d
  __int64 *v25; // r11
  char *v26; // r14
  bool v27; // zf
  char *v28; // rbx
  unsigned __int64 v29; // r15
  __int64 v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdx
  _QWORD *v34; // rax
  __int64 v36; // r12
  __int64 *v37; // r13
  unsigned __int64 v38; // rax
  __int64 *v39; // r12
  __int64 v40; // rbx
  unsigned int v41; // r13d
  __int64 v42; // [rsp+8h] [rbp-528h]
  __int64 *v43; // [rsp+10h] [rbp-520h]
  unsigned __int64 v45; // [rsp+28h] [rbp-508h]
  __int64 *v46; // [rsp+30h] [rbp-500h]
  __int64 *v47; // [rsp+30h] [rbp-500h]
  unsigned int v48; // [rsp+3Ch] [rbp-4F4h]
  __int64 v50; // [rsp+48h] [rbp-4E8h]
  unsigned int v51; // [rsp+50h] [rbp-4E0h]
  unsigned __int8 v52; // [rsp+58h] [rbp-4D8h]
  __int64 *v53; // [rsp+58h] [rbp-4D8h]
  __int64 *v54[4]; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 *v55[4]; // [rsp+80h] [rbp-4B0h] BYREF
  void *src; // [rsp+A0h] [rbp-490h] BYREF
  unsigned int v57; // [rsp+A8h] [rbp-488h]
  char v58; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD *v59; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v60; // [rsp+F8h] [rbp-438h]
  _QWORD v61[134]; // [rsp+100h] [rbp-430h] BYREF

  v6 = (__int64 *)a6;
  v59 = v61;
  v61[1] = (unsigned int)a5;
  v48 = a3;
  v61[0] = a2;
  v60 = 0x4000000001LL;
  *(_DWORD *)(sub_2EB5B40(a1, a2, a3, a4, a5, a6) + 4) = a5;
  for ( i = v60; (_DWORD)v60; i = v60 )
  {
    while ( 1 )
    {
      v10 = (__int64)&v59[2 * i - 2];
      v11 = *(_QWORD *)v10;
      v12 = *(_DWORD *)(v10 + 8);
      LODWORD(v60) = i - 1;
      v13 = sub_2EB5B40(a1, v11, v10, (__int64)v59, v7, v8);
      v14 = *(unsigned int *)(v13 + 32);
      v15 = *(unsigned int *)(v13 + 36);
      if ( v14 + 1 > v15 )
      {
        sub_C8D5F0(v13 + 24, (const void *)(v13 + 40), v14 + 1, 4u, v7, v8);
        v14 = *(unsigned int *)(v13 + 32);
      }
      v16 = *(_QWORD *)(v13 + 24);
      *(_DWORD *)(v16 + 4 * v14) = v12;
      v17 = *(_DWORD *)v13;
      ++*(_DWORD *)(v13 + 32);
      if ( !v17 )
      {
        LODWORD(v18) = a1;
        ++v48;
        *(_DWORD *)(v13 + 4) = v12;
        *(_DWORD *)(v13 + 12) = v48;
        *(_DWORD *)(v13 + 8) = v48;
        *(_DWORD *)v13 = v48;
        sub_2E6D5A0(a1, v11, v16, v15, v7, v8);
        sub_2EB52F0(&src, v11, *(_QWORD *)(a1 + 4128), v19, v20, v21);
        v22 = v57;
        if ( v6 && v57 > 1uLL )
        {
          v36 = 8LL * v57;
          v37 = (__int64 *)src;
          v18 = (__int64 *)((char *)src + v36);
          _BitScanReverse64(&v38, v36 >> 3);
          v43 = (__int64 *)((char *)src + v36);
          sub_2EBC160((__int64 *)src, (char *)src + v36, 2LL * (int)(63 - (v38 ^ 0x3F)), (__int64)v6);
          if ( (unsigned __int64)v36 <= 0x80 )
          {
            sub_2EB7D20(v37, v43, (__int64)v6);
          }
          else
          {
            v47 = v37 + 16;
            sub_2EB7D20(v37, v37 + 16, (__int64)v6);
            if ( v37 + 16 != v18 )
            {
              v42 = v11;
              do
              {
                v39 = v47;
                v40 = *v47;
                while ( 1 )
                {
                  v18 = (__int64 *)*(v39 - 1);
                  v53 = v39--;
                  sub_2E6E850(v54, v6, v40);
                  v41 = *((_DWORD *)v54[2] + 2);
                  sub_2E6E850(v55, v6, (__int64)v18);
                  if ( v41 >= *((_DWORD *)v55[2] + 2) )
                    break;
                  v39[1] = *v39;
                }
                ++v47;
                *v53 = v40;
              }
              while ( v47 != v43 );
              v11 = v42;
            }
          }
          v22 = v57;
        }
        v23 = (char *)src;
        if ( (char *)src + 8 * v22 != src )
        {
          v7 = a4;
          v24 = v48;
          v25 = v6;
          v26 = (char *)src + 8 * v22;
          v27 = v11 == a4;
          v28 = (char *)src;
          LOBYTE(v18) = !v27;
          v8 = (unsigned int)v18;
          v29 = v45;
          do
          {
            v30 = *(_QWORD *)v28;
            if ( *(_QWORD *)v28 != v7 && (_BYTE)v8 )
            {
              v31 = (unsigned int)v60;
              v32 = v29 & 0xFFFFFFFF00000000LL | v24;
              v33 = (unsigned int)v60 + 1LL;
              v29 = v32;
              if ( v33 > HIDWORD(v60) )
              {
                v46 = v25;
                v50 = v7;
                v51 = v24;
                v52 = v8;
                sub_C8D5F0((__int64)&v59, v61, v33, 0x10u, v7, v8);
                v31 = (unsigned int)v60;
                v25 = v46;
                v7 = v50;
                v24 = v51;
                v8 = v52;
              }
              v34 = &v59[2 * v31];
              *v34 = v30;
              v34[1] = v32;
              LODWORD(v60) = v60 + 1;
            }
            v28 += 8;
          }
          while ( v26 != v28 );
          v45 = v29;
          v23 = (char *)src;
          v6 = v25;
        }
        if ( v23 != &v58 )
          break;
      }
      i = v60;
      if ( !(_DWORD)v60 )
        goto LABEL_20;
    }
    _libc_free((unsigned __int64)v23);
  }
LABEL_20:
  if ( v59 != v61 )
    _libc_free((unsigned __int64)v59);
  return v48;
}

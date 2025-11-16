// Function: sub_2E755A0
// Address: 0x2e755a0
//
void __fastcall sub_2E755A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r15
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
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // rax
  char *v21; // r14
  __int64 *v22; // r10
  char *v23; // r15
  unsigned __int64 v24; // r14
  bool v25; // zf
  char *v26; // rbx
  bool v27; // al
  __int64 v28; // r13
  __int64 v29; // rdx
  unsigned __int64 v30; // r12
  _QWORD *v31; // rdx
  __int64 v32; // r12
  __int64 *v33; // r13
  __int64 *v34; // r14
  unsigned __int64 v35; // rax
  __int64 *v36; // r12
  __int64 v37; // rbx
  __int64 v38; // r14
  unsigned int v39; // r13d
  __int64 v40; // [rsp+8h] [rbp-528h]
  __int64 *v41; // [rsp+10h] [rbp-520h]
  unsigned __int64 v43; // [rsp+28h] [rbp-508h]
  __int64 *v44; // [rsp+30h] [rbp-500h]
  __int64 *v45; // [rsp+30h] [rbp-500h]
  unsigned int v46; // [rsp+3Ch] [rbp-4F4h]
  __int64 v47; // [rsp+48h] [rbp-4E8h]
  unsigned int v48; // [rsp+50h] [rbp-4E0h]
  bool v49; // [rsp+58h] [rbp-4D8h]
  __int64 *v50; // [rsp+58h] [rbp-4D8h]
  __int64 *v51[4]; // [rsp+60h] [rbp-4D0h] BYREF
  __int64 *v52[4]; // [rsp+80h] [rbp-4B0h] BYREF
  void *src; // [rsp+A0h] [rbp-490h] BYREF
  unsigned int v54; // [rsp+A8h] [rbp-488h]
  char v55; // [rsp+B0h] [rbp-480h] BYREF
  _QWORD *v56; // [rsp+F0h] [rbp-440h] BYREF
  __int64 v57; // [rsp+F8h] [rbp-438h]
  _QWORD v58[134]; // [rsp+100h] [rbp-430h] BYREF

  v6 = (__int64 *)a6;
  v56 = v58;
  v58[1] = (unsigned int)a5;
  v46 = a3;
  v58[0] = a2;
  v57 = 0x4000000001LL;
  *(_DWORD *)(sub_2E6F1C0(a1, a2, a3, a4, a5, a6) + 4) = a5;
  for ( i = v57; (_DWORD)v57; i = v57 )
  {
    while ( 1 )
    {
      v10 = (__int64)&v56[2 * i - 2];
      v11 = *(_QWORD *)v10;
      v12 = *(_DWORD *)(v10 + 8);
      LODWORD(v57) = i - 1;
      v13 = sub_2E6F1C0(a1, v11, v10, (__int64)v56, v7, v8);
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
        ++v46;
        *(_DWORD *)(v13 + 4) = v12;
        *(_DWORD *)(v13 + 12) = v46;
        *(_DWORD *)(v13 + 8) = v46;
        *(_DWORD *)v13 = v46;
        sub_2E6D5A0(a1, v11, v16, v15, v7, v8);
        sub_2E6EC80(&src, v11, *(_QWORD *)(a1 + 4128), v18, v19);
        v20 = v54;
        if ( v6 && v54 > 1uLL )
        {
          v32 = 8LL * v54;
          v33 = (__int64 *)src;
          v34 = (__int64 *)((char *)src + v32);
          _BitScanReverse64(&v35, v32 >> 3);
          v41 = (__int64 *)((char *)src + v32);
          sub_2E750C0((__int64 *)src, (char *)src + v32, 2LL * (int)(63 - (v35 ^ 0x3F)), (__int64)v6);
          if ( (unsigned __int64)v32 <= 0x80 )
          {
            sub_2E73D80(v33, v41, (__int64)v6);
          }
          else
          {
            v45 = v33 + 16;
            sub_2E73D80(v33, v33 + 16, (__int64)v6);
            if ( v33 + 16 != v34 )
            {
              v40 = v11;
              do
              {
                v36 = v45;
                v37 = *v45;
                while ( 1 )
                {
                  v38 = *(v36 - 1);
                  v50 = v36--;
                  sub_2E6E850(v51, v6, v37);
                  v39 = *((_DWORD *)v51[2] + 2);
                  sub_2E6E850(v52, v6, v38);
                  if ( v39 >= *((_DWORD *)v52[2] + 2) )
                    break;
                  v36[1] = *v36;
                }
                ++v45;
                *v50 = v37;
              }
              while ( v45 != v41 );
              v11 = v40;
            }
          }
          v20 = v54;
        }
        v21 = (char *)src + 8 * v20;
        if ( src != v21 )
        {
          v7 = a4;
          v22 = v6;
          v8 = v46;
          v23 = (char *)src + 8 * v20;
          v24 = v43;
          v25 = a4 == v11;
          v26 = (char *)src;
          v27 = !v25;
          do
          {
            v28 = *(_QWORD *)v26;
            if ( *(_QWORD *)v26 != v7 && v27 )
            {
              v29 = (unsigned int)v57;
              v30 = v24 & 0xFFFFFFFF00000000LL | (unsigned int)v8;
              v24 = v30;
              if ( (unsigned __int64)(unsigned int)v57 + 1 > HIDWORD(v57) )
              {
                v44 = v22;
                v47 = v7;
                v48 = v8;
                v49 = v27;
                sub_C8D5F0((__int64)&v56, v58, (unsigned int)v57 + 1LL, 0x10u, v7, v8);
                v29 = (unsigned int)v57;
                v22 = v44;
                v7 = v47;
                v8 = v48;
                v27 = v49;
              }
              v31 = &v56[2 * v29];
              *v31 = v28;
              v31[1] = v30;
              LODWORD(v57) = v57 + 1;
            }
            v26 += 8;
          }
          while ( v23 != v26 );
          v43 = v24;
          v21 = (char *)src;
          v6 = v22;
        }
        if ( v21 != &v55 )
          break;
      }
      i = v57;
      if ( !(_DWORD)v57 )
        goto LABEL_20;
    }
    _libc_free((unsigned __int64)v21);
  }
LABEL_20:
  if ( v56 != v58 )
    _libc_free((unsigned __int64)v56);
}

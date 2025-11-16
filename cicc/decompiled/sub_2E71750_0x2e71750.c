// Function: sub_2E71750
// Address: 0x2e71750
//
void __fastcall sub_2E71750(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // r15
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  unsigned int v12; // eax
  __int64 v13; // rax
  unsigned __int64 *v14; // r13
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // eax
  unsigned __int64 *v20; // r15
  __int64 v21; // rdx
  __int64 v22; // r12
  int v23; // r10d
  __int64 v24; // rax
  int v25; // r10d
  __int64 v26; // r14
  __int64 v27; // rax
  int v28; // eax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 *v31; // r14
  char *v32; // r10
  unsigned int v33; // eax
  __int64 v34; // rax
  unsigned __int64 v35; // r11
  _QWORD *v36; // rax
  __int64 v37; // r12
  unsigned __int64 *v38; // rax
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  _BYTE *v42; // rbx
  unsigned __int64 v43; // r12
  unsigned __int64 v44; // rdi
  __int64 v45; // [rsp+8h] [rbp-1508h]
  char *v46; // [rsp+10h] [rbp-1500h]
  unsigned int v47; // [rsp+1Ch] [rbp-14F4h]
  __int64 *v48; // [rsp+20h] [rbp-14F0h]
  unsigned int v49; // [rsp+40h] [rbp-14D0h]
  unsigned int v50; // [rsp+44h] [rbp-14CCh]
  int v51; // [rsp+48h] [rbp-14C8h]
  unsigned __int64 v52; // [rsp+48h] [rbp-14C8h]
  __int64 *v53; // [rsp+50h] [rbp-14C0h] BYREF
  int v54; // [rsp+58h] [rbp-14B8h]
  char v55; // [rsp+60h] [rbp-14B0h] BYREF
  _QWORD *v56; // [rsp+A0h] [rbp-1470h] BYREF
  __int64 v57; // [rsp+A8h] [rbp-1468h]
  _QWORD v58[128]; // [rsp+B0h] [rbp-1460h] BYREF
  unsigned __int64 v59[2]; // [rsp+4B0h] [rbp-1060h] BYREF
  _QWORD v60[64]; // [rsp+4C0h] [rbp-1050h] BYREF
  _BYTE *v61; // [rsp+6C0h] [rbp-E50h]
  __int64 v62; // [rsp+6C8h] [rbp-E48h]
  _BYTE v63[3584]; // [rsp+6D0h] [rbp-E40h] BYREF
  __int64 v64; // [rsp+14D0h] [rbp-40h]

  v5 = a1;
  v7 = sub_2E6D000(a1, a3, a4);
  v10 = (_QWORD *)v7;
  if ( v7 )
  {
    v11 = (unsigned int)(*(_DWORD *)(v7 + 24) + 1);
    v12 = *(_DWORD *)(v7 + 24) + 1;
  }
  else
  {
    v11 = 0;
    v12 = 0;
  }
  if ( v12 >= *(_DWORD *)(a1 + 32) )
    BUG();
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v11);
  v48 = *(__int64 **)(v13 + 8);
  if ( v48 )
  {
    v14 = v59;
    v49 = *(_DWORD *)(v13 + 16);
    v59[0] = (unsigned __int64)v60;
    v61 = v63;
    v62 = 0x4000000000LL;
    v64 = a2;
    v59[1] = 0x4000000001LL;
    v60[0] = 0;
    v56 = v58;
    v58[0] = v10;
    v58[1] = 0;
    v57 = 0x4000000001LL;
    *(_DWORD *)(sub_2E6F1C0((__int64)v59, (__int64)v10, v11, (__int64)v58, v8, v9) + 4) = 0;
    v19 = v57;
    if ( (_DWORD)v57 )
    {
      v50 = 0;
      v20 = v59;
      do
      {
        while ( 1 )
        {
          v21 = (__int64)&v56[2 * v19 - 2];
          v22 = *(_QWORD *)v21;
          v23 = *(_DWORD *)(v21 + 8);
          LODWORD(v57) = v19 - 1;
          v10 = (_QWORD *)v22;
          v51 = v23;
          v24 = sub_2E6F1C0((__int64)v20, v22, v21, (__int64)v56, v17, v18);
          v25 = v51;
          v26 = v24;
          v27 = *(unsigned int *)(v24 + 32);
          v16 = *(unsigned int *)(v26 + 36);
          if ( v27 + 1 > v16 )
          {
            v10 = (_QWORD *)(v26 + 40);
            sub_C8D5F0(v26 + 24, (const void *)(v26 + 40), v27 + 1, 4u, v17, v18);
            v27 = *(unsigned int *)(v26 + 32);
            v25 = v51;
          }
          v15 = *(_QWORD *)(v26 + 24);
          *(_DWORD *)(v15 + 4 * v27) = v25;
          v28 = *(_DWORD *)v26;
          ++*(_DWORD *)(v26 + 32);
          if ( !v28 )
          {
            ++v50;
            *(_DWORD *)(v26 + 4) = v25;
            *(_DWORD *)(v26 + 12) = v50;
            *(_DWORD *)(v26 + 8) = v50;
            *(_DWORD *)v26 = v50;
            sub_2E6D5A0((__int64)v20, v22, v15, v16, v17, v18);
            v10 = (_QWORD *)v22;
            sub_2E6EC80(&v53, v22, v64, v29, v30);
            v31 = v53;
            v32 = (char *)&v53[v54];
            if ( v53 != (__int64 *)v32 )
            {
              v18 = (__int64)v20;
              v17 = v50;
              do
              {
                v37 = *v31;
                if ( *v31 )
                {
                  v15 = (unsigned int)(*(_DWORD *)(v37 + 24) + 1);
                  v33 = *(_DWORD *)(v37 + 24) + 1;
                }
                else
                {
                  v33 = 0;
                  v15 = 0;
                }
                if ( v33 >= *(_DWORD *)(a1 + 32) )
                  BUG();
                if ( v49 < *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 8 * v15) + 16LL) )
                {
                  v16 = HIDWORD(v57);
                  v34 = (unsigned int)v57;
                  v35 = v4 & 0xFFFFFFFF00000000LL | (unsigned int)v17;
                  v15 = (unsigned int)v57 + 1LL;
                  v4 = v35;
                  if ( v15 > HIDWORD(v57) )
                  {
                    v10 = v58;
                    v45 = v18;
                    v46 = v32;
                    v47 = v17;
                    v52 = v35;
                    sub_C8D5F0((__int64)&v56, v58, v15, 0x10u, v17, v18);
                    v34 = (unsigned int)v57;
                    v18 = v45;
                    v32 = v46;
                    v17 = v47;
                    v35 = v52;
                  }
                  v36 = &v56[2 * v34];
                  *v36 = v37;
                  v36[1] = v35;
                  LODWORD(v57) = v57 + 1;
                }
                ++v31;
              }
              while ( v32 != (char *)v31 );
              v32 = (char *)v53;
              v20 = (unsigned __int64 *)v18;
            }
            if ( v32 != &v55 )
              break;
          }
          v19 = v57;
          if ( !(_DWORD)v57 )
            goto LABEL_25;
        }
        _libc_free((unsigned __int64)v32);
        v19 = v57;
      }
      while ( (_DWORD)v57 );
LABEL_25:
      v38 = v20;
      v5 = a1;
      v14 = v38;
    }
    if ( v56 != v58 )
      _libc_free((unsigned __int64)v56);
    sub_2E6F370((__int64)v14, (__int64)v10, v15, v16, v17, v18);
    sub_2E6F7A0(v14, v5, *v48, v39, v40, v41);
    v42 = v61;
    v43 = (unsigned __int64)&v61[56 * (unsigned int)v62];
    if ( v61 != (_BYTE *)v43 )
    {
      do
      {
        v43 -= 56LL;
        v44 = *(_QWORD *)(v43 + 24);
        if ( v44 != v43 + 40 )
          _libc_free(v44);
      }
      while ( v42 != (_BYTE *)v43 );
      v43 = (unsigned __int64)v61;
    }
    if ( (_BYTE *)v43 != v63 )
      _libc_free(v43);
    if ( (_QWORD *)v59[0] != v60 )
      _libc_free(v59[0]);
  }
  else
  {
    sub_2E70350(a1, a2);
  }
}

// Function: sub_16750C0
// Address: 0x16750c0
//
void __fastcall sub_16750C0(__int64 a1)
{
  __int64 v2; // r14
  __int64 v3; // rdx
  __int64 *v4; // rax
  unsigned int v5; // esi
  __int64 v6; // r13
  __int64 v7; // r8
  unsigned int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  __int64 v12; // rax
  unsigned int v13; // ebx
  const void *v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r14
  unsigned __int64 v17; // rbx
  __int64 *v18; // rbx
  _QWORD **v19; // rsi
  __int64 v20; // rax
  void *v21; // rdi
  unsigned int v22; // eax
  __int64 v23; // rdx
  _QWORD *v24; // rax
  _QWORD *v25; // rdx
  int v26; // r10d
  _QWORD *v27; // rdx
  int v28; // eax
  int v29; // ecx
  int v30; // eax
  int v31; // edi
  __int64 v32; // rsi
  unsigned int v33; // eax
  __int64 v34; // r8
  int v35; // r10d
  _QWORD *v36; // r9
  int v37; // eax
  int v38; // eax
  __int64 v39; // rdi
  _QWORD *v40; // r8
  unsigned int v41; // ebx
  int v42; // r9d
  __int64 v43; // rsi
  __int64 *v44; // r11
  __int64 v45; // [rsp+0h] [rbp-170h]
  __int64 *v46; // [rsp+10h] [rbp-160h]
  __int64 v47; // [rsp+18h] [rbp-158h]
  __int64 *v48; // [rsp+20h] [rbp-150h]
  __int64 v49; // [rsp+28h] [rbp-148h]
  __int64 v50; // [rsp+38h] [rbp-138h]
  __int64 v51; // [rsp+40h] [rbp-130h] BYREF
  _BYTE *v52; // [rsp+48h] [rbp-128h]
  _BYTE *v53; // [rsp+50h] [rbp-120h]
  __int64 v54; // [rsp+58h] [rbp-118h]
  int v55; // [rsp+60h] [rbp-110h]
  _BYTE v56[72]; // [rsp+68h] [rbp-108h] BYREF
  _BYTE *v57; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+B8h] [rbp-B8h]
  _BYTE v59[176]; // [rsp+C0h] [rbp-B0h] BYREF

  v2 = a1;
  v3 = *(unsigned int *)(a1 + 336);
  v57 = v59;
  v58 = 0x1000000000LL;
  v4 = *(__int64 **)(a1 + 328);
  v48 = v4;
  v46 = &v4[v3];
  v45 = a1 + 8;
  if ( v4 != v46 )
  {
    while ( 1 )
    {
      v5 = *(_DWORD *)(a1 + 32);
      v6 = *v48;
      if ( v5 )
      {
        v7 = *(_QWORD *)(a1 + 16);
        v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v9 = (_QWORD *)(v7 + 16LL * v8);
        v10 = *v9;
        if ( v6 == *v9 )
        {
          v47 = v9[1];
          goto LABEL_5;
        }
        v26 = 1;
        v27 = 0;
        while ( v10 != -8 )
        {
          if ( v27 || v10 != -16 )
            v9 = v27;
          v8 = (v5 - 1) & (v26 + v8);
          v44 = (__int64 *)(v7 + 16LL * v8);
          v10 = *v44;
          if ( v6 == *v44 )
          {
            v47 = v44[1];
            goto LABEL_5;
          }
          ++v26;
          v27 = v9;
          v9 = (_QWORD *)(v7 + 16LL * v8);
        }
        if ( !v27 )
          v27 = v9;
        v28 = *(_DWORD *)(a1 + 24);
        ++*(_QWORD *)(a1 + 8);
        v29 = v28 + 1;
        if ( 4 * (v28 + 1) < 3 * v5 )
        {
          if ( v5 - *(_DWORD *)(a1 + 28) - v29 <= v5 >> 3 )
          {
            sub_1670A20(v45, v5);
            v37 = *(_DWORD *)(a1 + 32);
            if ( !v37 )
            {
LABEL_75:
              ++*(_DWORD *)(a1 + 24);
              BUG();
            }
            v38 = v37 - 1;
            v39 = *(_QWORD *)(a1 + 16);
            v40 = 0;
            v41 = v38 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
            v42 = 1;
            v29 = *(_DWORD *)(a1 + 24) + 1;
            v27 = (_QWORD *)(v39 + 16LL * v41);
            v43 = *v27;
            if ( v6 != *v27 )
            {
              while ( v43 != -8 )
              {
                if ( !v40 && v43 == -16 )
                  v40 = v27;
                v41 = v38 & (v42 + v41);
                v27 = (_QWORD *)(v39 + 16LL * v41);
                v43 = *v27;
                if ( v6 == *v27 )
                  goto LABEL_41;
                ++v42;
              }
              if ( v40 )
                v27 = v40;
            }
          }
          goto LABEL_41;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      sub_1670A20(v45, 2 * v5);
      v30 = *(_DWORD *)(a1 + 32);
      if ( !v30 )
        goto LABEL_75;
      v31 = v30 - 1;
      v32 = *(_QWORD *)(a1 + 16);
      v33 = (v30 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v29 = *(_DWORD *)(a1 + 24) + 1;
      v27 = (_QWORD *)(v32 + 16LL * v33);
      v34 = *v27;
      if ( v6 != *v27 )
      {
        v35 = 1;
        v36 = 0;
        while ( v34 != -8 )
        {
          if ( !v36 && v34 == -16 )
            v36 = v27;
          v33 = v31 & (v35 + v33);
          v27 = (_QWORD *)(v32 + 16LL * v33);
          v34 = *v27;
          if ( v6 == *v27 )
            goto LABEL_41;
          ++v35;
        }
        if ( v36 )
          v27 = v36;
      }
LABEL_41:
      *(_DWORD *)(a1 + 24) = v29;
      if ( *v27 != -8 )
        --*(_DWORD *)(a1 + 28);
      *v27 = v6;
      v27[1] = 0;
      v47 = 0;
LABEL_5:
      v11 = *(unsigned int *)(v6 + 12);
      v12 = (unsigned int)v58;
      v13 = *(_DWORD *)(v6 + 12);
      if ( v11 >= (unsigned int)v58 )
      {
        if ( v11 > (unsigned int)v58 )
        {
          if ( v11 > HIDWORD(v58) )
          {
            sub_16CD150(&v57, v59, *(unsigned int *)(v6 + 12), 8);
            v12 = (unsigned int)v58;
          }
          v14 = v57;
          v24 = &v57[8 * v12];
          v25 = &v57[8 * v11];
          if ( v24 != v25 )
          {
            do
            {
              if ( v24 )
                *v24 = 0;
              ++v24;
            }
            while ( v25 != v24 );
            v14 = v57;
          }
          LODWORD(v58) = v11;
        }
        else
        {
          v14 = v57;
          v13 = v58;
        }
      }
      else
      {
        LODWORD(v58) = *(_DWORD *)(v6 + 12);
        v14 = v57;
      }
      if ( v13 )
      {
        v15 = v13 - 1;
        v16 = 0;
        v17 = (unsigned __int64)v14;
        v50 = 8 * v15;
        while ( 1 )
        {
          v18 = (__int64 *)(v16 + v17);
          v19 = *(_QWORD ***)(*(_QWORD *)(v6 + 16) + v16);
          v52 = v56;
          v51 = 0;
          v53 = v56;
          v54 = 8;
          v55 = 0;
          v20 = sub_1674800(a1, v19, (__int64)&v51);
          if ( v53 != v52 )
          {
            v49 = v20;
            _libc_free((unsigned __int64)v53);
            v20 = v49;
          }
          *v18 = v20;
          if ( v50 == v16 )
            break;
          v17 = (unsigned __int64)v57;
          v16 += 8;
        }
        v14 = v57;
        v13 = v58;
      }
      sub_1643FB0(v47, v14, v13, (*(_DWORD *)(v6 + 8) & 0x200) != 0);
      sub_1673870(*(_QWORD *)(a1 + 640), v47);
      if ( v46 == ++v48 )
      {
        v2 = a1;
        break;
      }
    }
  }
  ++*(_QWORD *)(v2 + 472);
  v21 = *(void **)(v2 + 488);
  *(_DWORD *)(v2 + 336) = 0;
  if ( v21 == *(void **)(v2 + 480) )
    goto LABEL_21;
  v22 = 4 * (*(_DWORD *)(v2 + 500) - *(_DWORD *)(v2 + 504));
  v23 = *(unsigned int *)(v2 + 496);
  if ( v22 < 0x20 )
    v22 = 32;
  if ( v22 >= (unsigned int)v23 )
  {
    memset(v21, -1, 8 * v23);
LABEL_21:
    *(_QWORD *)(v2 + 500) = 0;
    goto LABEL_22;
  }
  sub_16CC920(v2 + 472);
LABEL_22:
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
}

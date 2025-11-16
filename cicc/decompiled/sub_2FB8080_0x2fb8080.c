// Function: sub_2FB8080
// Address: 0x2fb8080
//
void __fastcall sub_2FB8080(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  unsigned int v5; // ebx
  int v6; // r14d
  unsigned int v7; // esi
  _QWORD *v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  unsigned int v11; // eax
  unsigned int v12; // r15d
  __int64 v13; // r13
  int v14; // ebx
  unsigned int v15; // esi
  __int64 v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // rbx
  __int64 v22; // rdx
  signed __int64 v23; // r13
  __int64 v24; // r15
  __int64 *v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int64 v28; // rcx
  __int64 *v29; // rax
  char v30; // dl
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rdx
  _QWORD *v34; // rdi
  __int64 v35; // rsi
  __int64 *v36; // rcx
  __int64 v37; // [rsp+10h] [rbp-D0h]
  __int64 v38; // [rsp+18h] [rbp-C8h]
  _QWORD *v39; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v40; // [rsp+28h] [rbp-B8h]
  _QWORD v41[4]; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v42; // [rsp+50h] [rbp-90h] BYREF
  __int64 *v43; // [rsp+58h] [rbp-88h]
  __int64 v44; // [rsp+60h] [rbp-80h]
  int v45; // [rsp+68h] [rbp-78h]
  char v46; // [rsp+6Ch] [rbp-74h]
  __int64 v47; // [rsp+70h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a1 + 72);
  if ( (*(_BYTE *)(a2 + 8) & 6) != 0 )
  {
    v5 = 0;
    v6 = *(_DWORD *)(*(_QWORD *)(v4 + 16) + 8LL) - *(_DWORD *)(v4 + 64);
    if ( v6 )
    {
      do
      {
        v7 = v5++;
        sub_2FB7E60(a1, v7, (int *)a2);
      }
      while ( v6 != v5 );
    }
    return;
  }
  v47 = a2;
  v8 = v41;
  v43 = &v47;
  v44 = 0x100000008LL;
  v40 = 0x400000001LL;
  v9 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v39 = v41;
  v42 = 1;
  v45 = 0;
  v46 = 1;
  v41[0] = a2;
  v10 = *(_QWORD *)(v4 + 8);
  v37 = v9;
  v11 = 1;
  while ( 1 )
  {
    v12 = 0;
    v13 = v8[v11 - 1];
    LODWORD(v40) = v11 - 1;
    v14 = *(_DWORD *)(*(_QWORD *)(v4 + 16) + 8LL) - *(_DWORD *)(v4 + 64);
    if ( v14 )
    {
      do
      {
        v15 = v12++;
        sub_2FB7E60(a1, v15, (int *)v13);
      }
      while ( v14 != v12 );
    }
    v16 = *(_QWORD *)(v13 + 8);
    if ( (v16 & 6) == 0 )
    {
      v17 = v16 & 0xFFFFFFFFFFFFFFF8LL;
      v18 = *(_QWORD *)(v17 + 16);
      if ( v18 )
      {
        v19 = *(_QWORD *)(v18 + 24);
      }
      else
      {
        v33 = *(unsigned int *)(v37 + 304);
        v34 = *(_QWORD **)(v37 + 296);
        if ( *(_DWORD *)(v37 + 304) )
        {
          do
          {
            while ( 1 )
            {
              v35 = v33 >> 1;
              v36 = &v34[2 * (v33 >> 1)];
              if ( *(_DWORD *)(v17 + 24) < (*(_DWORD *)((*v36 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                          | (unsigned int)(*v36 >> 1) & 3) )
                break;
              v34 = v36 + 2;
              v33 = v33 - v35 - 1;
              if ( v33 <= 0 )
                goto LABEL_44;
            }
            v33 >>= 1;
          }
          while ( v35 > 0 );
        }
LABEL_44:
        v19 = *(v34 - 1);
      }
      v20 = *(_QWORD *)(v19 + 64);
      v38 = v20 + 8LL * *(unsigned int *)(v19 + 72);
      if ( v20 != v38 )
      {
        v21 = *(_QWORD *)(v19 + 64);
        while ( 1 )
        {
          v22 = *(_QWORD *)(*(_QWORD *)(v37 + 152) + 16LL * *(unsigned int *)(*(_QWORD *)v21 + 24LL) + 8);
          if ( ((v22 >> 1) & 3) != 0 )
            v23 = v22 & 0xFFFFFFFFFFFFFFF8LL | (2LL * (int)(((v22 >> 1) & 3) - 1));
          else
            v23 = *(_QWORD *)(v22 & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL | 6;
          v24 = 0;
          v25 = (__int64 *)sub_2E09D00((__int64 *)v10, v23);
          v28 = 3LL * *(unsigned int *)(v10 + 8);
          if ( v25 != (__int64 *)(*(_QWORD *)v10 + 24LL * *(unsigned int *)(v10 + 8)) )
          {
            v28 = v23 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_DWORD *)((*v25 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v25 >> 1) & 3) <= (*(_DWORD *)((v23 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v23 >> 1) & 3) )
              v24 = v25[2];
          }
          if ( v46 )
          {
            v29 = v43;
            v28 = HIDWORD(v44);
            v25 = &v43[HIDWORD(v44)];
            if ( v43 != v25 )
            {
              while ( v24 != *v29 )
              {
                if ( v25 == ++v29 )
                  goto LABEL_33;
              }
              goto LABEL_25;
            }
LABEL_33:
            if ( HIDWORD(v44) < (unsigned int)v44 )
            {
              ++HIDWORD(v44);
              *v25 = v24;
              ++v42;
              goto LABEL_28;
            }
          }
          sub_C8CC70((__int64)&v42, v24, (__int64)v25, v28, v26, v27);
          if ( v30 )
          {
LABEL_28:
            v31 = (unsigned int)v40;
            v32 = (unsigned int)v40 + 1LL;
            if ( v32 > HIDWORD(v40) )
            {
              sub_C8D5F0((__int64)&v39, v41, v32, 8u, v26, v27);
              v31 = (unsigned int)v40;
            }
            v21 += 8;
            v39[v31] = v24;
            LODWORD(v40) = v40 + 1;
            if ( v38 == v21 )
              break;
          }
          else
          {
LABEL_25:
            v21 += 8;
            if ( v38 == v21 )
              break;
          }
        }
      }
    }
    v11 = v40;
    v8 = v39;
    if ( !(_DWORD)v40 )
      break;
    v4 = *(_QWORD *)(a1 + 72);
  }
  if ( v39 != v41 )
    _libc_free((unsigned __int64)v39);
  if ( !v46 )
    _libc_free((unsigned __int64)v43);
}

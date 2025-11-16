// Function: sub_1EBC180
// Address: 0x1ebc180
//
void __fastcall sub_1EBC180(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  _QWORD *v4; // r8
  __int64 v5; // r10
  void *v6; // rbx
  __int64 v7; // r13
  int v8; // eax
  __int64 v9; // r9
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // rsi
  unsigned int *v13; // rax
  int v14; // edx
  __int64 v15; // r11
  __int64 v16; // rdx
  int *v17; // r12
  __int64 v18; // r15
  int v19; // r13d
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // r13
  unsigned int v23; // r15d
  __int64 v24; // rax
  int v25; // r12d
  _QWORD *v26; // r14
  __int64 v27; // rdx
  __int64 v28; // r10
  __int64 v29; // rdi
  _QWORD *v30; // rdi
  __int64 *v31; // rsi
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 *v34; // rbx
  _QWORD *v35; // r8
  _QWORD *v36; // r8
  unsigned __int64 v37; // r12
  __int64 v38; // rax
  _QWORD *v39; // r8
  __int64 v40; // r10
  __int64 v41; // rax
  __int64 v42; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v43; // [rsp+10h] [rbp-D0h]
  unsigned int v44; // [rsp+28h] [rbp-B8h]
  const void *v45; // [rsp+30h] [rbp-B0h]
  unsigned int v46; // [rsp+38h] [rbp-A8h]
  _QWORD *v47; // [rsp+38h] [rbp-A8h]
  _QWORD *v48; // [rsp+38h] [rbp-A8h]
  __int64 v49; // [rsp+40h] [rbp-A0h]
  __int64 v50; // [rsp+40h] [rbp-A0h]
  __int64 v51; // [rsp+40h] [rbp-A0h]
  __int64 v52; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v53; // [rsp+48h] [rbp-98h]
  unsigned int *v54; // [rsp+48h] [rbp-98h]
  _QWORD *v55; // [rsp+48h] [rbp-98h]
  _QWORD *v56; // [rsp+48h] [rbp-98h]
  _QWORD *v57; // [rsp+48h] [rbp-98h]
  _QWORD *v58; // [rsp+48h] [rbp-98h]
  _QWORD *v59; // [rsp+48h] [rbp-98h]
  _QWORD *v60; // [rsp+48h] [rbp-98h]
  _DWORD v61[8]; // [rsp+50h] [rbp-90h] BYREF
  _DWORD v62[28]; // [rsp+70h] [rbp-70h] BYREF

  v4 = a1;
  v5 = a2;
  v6 = 0;
  v7 = a1[123];
  v8 = *(_DWORD *)(v7 + 640);
  if ( v8 )
  {
    v37 = 8LL * ((unsigned int)(v8 + 63) >> 6);
    v38 = malloc(v37);
    v39 = a1;
    v40 = a2;
    v6 = (void *)v38;
    if ( !v38 )
    {
      if ( v37 || (v41 = malloc(1u), v39 = a1, v40 = a2, !v41) )
      {
        v52 = v40;
        v60 = v39;
        sub_16BD1C0("Allocation failed", 1u);
        v39 = v60;
        v40 = v52;
      }
      else
      {
        v6 = (void *)v41;
      }
    }
    v51 = v40;
    v58 = v39;
    memcpy(v6, *(const void **)(v7 + 624), v37);
    v5 = v51;
    v4 = v58;
  }
  v43 = (unsigned __int64)v6;
  v9 = 1;
  v10 = v5;
  v44 = 0;
  v45 = (const void *)(v5 + 64);
  while ( 1 )
  {
    v11 = v4[106];
    v12 = *(unsigned int *)(v10 + 56);
    v13 = *(unsigned int **)(v11 + 328);
    v14 = *(_DWORD *)(v11 + 336);
    if ( v14 )
    {
      v15 = (__int64)&v13[v14 - 1 + 1];
      do
      {
        a4 = (_QWORD *)v4[105];
        v16 = a4[37] + 48LL * *v13;
        v17 = *(int **)v16;
        v18 = *(_QWORD *)v16 + 4LL * *(unsigned int *)(v16 + 8);
        if ( v18 != *(_QWORD *)v16 )
        {
          do
          {
            v19 = *v17;
            v20 = 1LL << *v17;
            a4 = (_QWORD *)(v43 + 8LL * ((unsigned int)*v17 >> 6));
            if ( (*a4 & v20) != 0 )
            {
              *a4 &= ~v20;
              if ( *(_DWORD *)(v10 + 60) <= (unsigned int)v12 )
              {
                v47 = v4;
                v50 = v15;
                v54 = v13;
                sub_16CD150(v10 + 48, v45, 0, 4, (int)v4, 1);
                v12 = *(unsigned int *)(v10 + 56);
                v4 = v47;
                v9 = 1;
                v15 = v50;
                v13 = v54;
              }
              *(_DWORD *)(*(_QWORD *)(v10 + 48) + 4 * v12) = v19;
              v12 = (unsigned int)(*(_DWORD *)(v10 + 56) + 1);
              *(_DWORD *)(v10 + 56) = v12;
            }
            ++v17;
          }
          while ( v17 != (int *)v18 );
        }
        ++v13;
      }
      while ( (unsigned int *)v15 != v13 );
    }
    if ( v44 == (_DWORD)v12 )
      break;
    v21 = *(_QWORD *)(v10 + 48);
    v53 = (unsigned int)v12 - (unsigned __int64)v44;
    v49 = v21 + 4LL * v44;
    if ( !*(_DWORD *)v10 )
    {
      v48 = v4;
      sub_1F12880(v4[106], v49, v53, 1, v4, 1);
      v35 = v48;
      goto LABEL_30;
    }
    v22 = *(_QWORD *)(v10 + 8);
    if ( v22 )
    {
      ++*(_DWORD *)(v22 + 8);
      if ( (unsigned int)v12 == (unsigned __int64)v44 )
      {
        v59 = v4;
        sub_1F12700(v4[106], v62, 0, a4, v4, 1);
        sub_1F12980(v59[106], v61, 0);
        v35 = v59;
LABEL_32:
        --*(_DWORD *)(v22 + 8);
        goto LABEL_30;
      }
    }
    else if ( !v53 )
    {
      v57 = v4;
      sub_1F12700(v4[106], v62, 0, a4, v4, 1);
      sub_1F12980(v57[106], v61, 0);
      v35 = v57;
      goto LABEL_30;
    }
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v42 = v10;
    v26 = v4;
    v46 = 0;
    do
    {
      v33 = *(unsigned int *)(v49 + 4 * v24);
      v34 = &qword_4FCF930;
      if ( v22 )
      {
        v34 = (__int64 *)(*(_QWORD *)(v22 + 512) + 24LL * (unsigned int)v33);
        if ( *(_DWORD *)v34 != *(_DWORD *)(v22 + 4) )
        {
          sub_20F85B0(v22, v33, v21, a4, v4, v9);
          v33 = (unsigned int)v33;
          v34 = (__int64 *)(*(_QWORD *)(v22 + 512) + 24LL * (unsigned int)v33);
        }
      }
      if ( (v34[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v27 = v26[100];
        v28 = v23;
        v62[2 * v23] = v33;
        v29 = *(_QWORD *)(*(_QWORD *)(v27 + 392) + 16 * v33);
        LODWORD(v27) = *(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v29 >> 1) & 3;
        v30 = (_QWORD *)v26[123];
        LOBYTE(v62[2 * v23 + 1]) = (unsigned int)v27 < (*(_DWORD *)((v34[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                      | (unsigned int)(v34[1] >> 1) & 3)
                                 ? 2
                                 : 4;
        v31 = (__int64 *)(v30[7] + 16LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(*v30 + 96LL) + 8 * v33) + 48LL));
        v32 = *v31;
        if ( (*v31 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v31[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v32 = sub_1F13A50(v30 + 6, v30[5]);
          v28 = v23;
        }
        v21 = *(_DWORD *)((v34[2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v34[2] >> 1) & 3;
        ++v23;
        BYTE1(v62[2 * v28 + 1]) = (unsigned int)v21 < (*(_DWORD *)((v32 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                     | (unsigned int)(v32 >> 1) & 3)
                                ? 2
                                : 4;
        if ( v23 == 8 )
        {
          v23 = 0;
          sub_1F12700(v26[106], v62, 8, a4, v4, v9);
        }
      }
      else
      {
        v61[v46++] = v33;
        if ( v46 == 8 )
        {
          sub_1F12980(v26[106], v61, 8);
          v46 = 0;
        }
      }
      v24 = (unsigned int)++v25;
    }
    while ( v53 != v25 );
    v36 = v26;
    v10 = v42;
    v56 = v36;
    sub_1F12700(v36[106], v62, v23, a4, v36, v9);
    sub_1F12980(v56[106], v61, v46);
    v35 = v56;
    if ( v22 )
      goto LABEL_32;
LABEL_30:
    v55 = v35;
    v44 = *(_DWORD *)(v10 + 56);
    sub_1F12FE0(v35[106]);
    v4 = v55;
    v9 = 1;
  }
  _libc_free(v43);
}

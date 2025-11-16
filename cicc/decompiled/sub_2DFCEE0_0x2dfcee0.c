// Function: sub_2DFCEE0
// Address: 0x2dfcee0
//
void __fastcall sub_2DFCEE0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r15
  _QWORD *v5; // r12
  __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // r9
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // r14
  _QWORD *v12; // rdx
  _QWORD *v13; // rax
  __int64 v14; // r8
  int v15; // eax
  unsigned __int64 *v16; // r13
  unsigned __int64 v17; // rbx
  unsigned int v18; // r15d
  __int64 v19; // r14
  __int64 v20; // rdx
  _QWORD *v21; // rax
  __int64 v22; // rax
  unsigned __int8 v23; // di
  __int64 v24; // rdx
  char v25; // al
  _QWORD *v26; // rdi
  char v27; // al
  _QWORD *v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 *v30; // rax
  __int64 v31; // rax
  void *v32; // rdi
  size_t v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // rdi
  const void **v36; // rsi
  void *v37; // r11
  size_t v38; // rdx
  unsigned __int64 v39; // rcx
  __int64 v40; // [rsp+8h] [rbp-C8h]
  unsigned __int64 v41; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+20h] [rbp-B0h]
  unsigned int v44; // [rsp+2Ch] [rbp-A4h]
  __int64 v45; // [rsp+30h] [rbp-A0h]
  __int64 v46; // [rsp+30h] [rbp-A0h]
  __int64 v49; // [rsp+58h] [rbp-78h]
  unsigned int v51; // [rsp+68h] [rbp-68h]
  int v52; // [rsp+6Ch] [rbp-64h]
  _DWORD v53[2]; // [rsp+78h] [rbp-58h] BYREF
  void *dest; // [rsp+80h] [rbp-50h] BYREF
  __int64 v55; // [rsp+88h] [rbp-48h]
  __int64 v56; // [rsp+90h] [rbp-40h]

  v4 = a1;
  v5 = *(_QWORD **)a1;
  v51 = *(_DWORD *)(*(_QWORD *)a1 + 160LL);
  if ( v51 )
  {
    sub_2DF52D0((__int64)&dest, a4);
  }
  else
  {
    dest = 0;
    v6 = *(_QWORD *)(a4 + 16);
    LOBYTE(v55) = *(_BYTE *)(a4 + 8);
    v56 = v6;
    if ( (v55 & 0x3F) != 0 )
    {
      v32 = (void *)sub_2207820(4 * (v55 & 0x3F));
      dest = v32;
      v33 = 4LL * (*(_BYTE *)(a4 + 8) & 0x3F);
      if ( v33 )
        memmove(v32, *(const void **)a4, v33);
    }
    v7 = sub_2DF7D50(
           (__int64)v5,
           (unsigned int *)(*(_QWORD *)(v4 + 8) + 16LL * *(unsigned int *)(v4 + 16) - 4),
           *((_DWORD *)v5 + 41),
           a2,
           a3,
           (__int64)&dest);
    if ( dest )
      j_j___libc_free_0_0((unsigned __int64)dest);
    if ( v7 <= 4 )
    {
      *((_DWORD *)v5 + 41) = v7;
      *(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) = v7;
      return;
    }
    v8 = sub_F03E60(
           2u,
           *((_DWORD *)v5 + 41),
           4,
           0,
           (__int64)v53,
           *(_DWORD *)(*(_QWORD *)(v4 + 8) + 16LL * *(unsigned int *)(v4 + 16) - 4),
           1u);
    dest = 0;
    v55 = 0;
    v41 = v8;
    v10 = (unsigned __int64 *)v5[21];
    v49 = 0;
    v40 = v4;
    v11 = *v10;
    if ( !*v10 )
      goto LABEL_38;
LABEL_9:
    *v10 = *(_QWORD *)v11;
LABEL_10:
    v12 = (_QWORD *)v11;
    memset((void *)v11, 0, 0xA0u);
    v13 = (_QWORD *)(v11 + 64);
    do
    {
      *v12 = 0;
      v12 += 2;
      *(v12 - 1) = 0;
    }
    while ( v13 != v12 );
    do
    {
      *v13 = 0;
      v13 += 3;
      *((_BYTE *)v13 - 16) = 0;
      *(v13 - 1) = 0;
    }
    while ( (_QWORD *)(v11 + 160) != v13 );
    v42 = v11 & 0xFFFFFFFFFFFFFFC0LL;
    while ( 1 )
    {
      v14 = v51;
      v15 = v53[v49];
      if ( v15 + v51 != v51 )
      {
        v52 = v15 + v51;
        v16 = (unsigned __int64 *)(v11 + 64);
        v17 = v11;
        v18 = v51;
        v19 = 0;
        do
        {
          v14 = v18;
          v9 = (unsigned int)v19;
          v20 = 16 * v19;
          v21 = &v5[2 * v18];
          *(_QWORD *)(v17 + v20) = *v21;
          *(_QWORD *)(v17 + v20 + 8) = v21[1];
          v22 = 3LL * v18;
          if ( &v5[v22 + 8] != v16 )
          {
            if ( (v5[v22 + 9] & 0x3F) != 0 )
            {
              v45 = (__int64)&v5[v22 + 8];
              v34 = sub_2207820(4LL * (v5[v22 + 9] & 0x3F));
              v35 = *v16;
              v36 = (const void **)v45;
              *v16 = v34;
              v9 = (unsigned int)v19;
              v37 = (void *)v34;
              v14 = v18;
              if ( v35 )
              {
                j_j___libc_free_0_0(v35);
                v37 = (void *)*v16;
                v14 = v18;
                v9 = (unsigned int)v19;
                v36 = (const void **)v45;
              }
              v46 = (__int64)&v5[3 * v14 + 8];
              v23 = v5[3 * v14 + 9] & 0x3F;
              v38 = 4LL * v23;
              if ( v38 )
              {
                v43 = v14;
                v44 = v9;
                memmove(v37, *v36, v38);
                v9 = v44;
                v14 = v43;
                v23 = *(_BYTE *)(v46 + 8) & 0x3F;
              }
            }
            else
            {
              *v16 = 0;
              v23 = v5[v22 + 9] & 0x3F;
            }
            v24 = v17 + 24 * v9 + 64;
            v25 = v23 | *(_BYTE *)(v17 + 24 * v9 + 72) & 0xC0;
            v26 = &v5[3 * v14];
            *(_BYTE *)(v24 + 8) = v25;
            v27 = v26[9] & 0x40 | v25 & 0xBF;
            *(_BYTE *)(v24 + 8) = v27;
            *(_BYTE *)(v24 + 8) = v26[9] & 0x80 | v27 & 0x7F;
            v16[2] = v26[10];
          }
          ++v18;
          ++v19;
          v16 += 3;
        }
        while ( v52 != v18 );
        v15 = v53[v49];
        v51 += v15;
      }
      *(&dest + v49) = (void *)(v42 | (unsigned int)(v15 - 1));
      if ( v49 == 1 )
        break;
      v10 = (unsigned __int64 *)v5[21];
      v49 = 1;
      v11 = *v10;
      if ( *v10 )
        goto LABEL_9;
LABEL_38:
      v39 = v10[1];
      v10[11] += 192LL;
      v42 = (v39 + 63) & 0xFFFFFFFFFFFFFFC0LL;
      if ( v10[2] < v42 + 192 || !v39 )
      {
        v11 = sub_9D1E70((__int64)(v10 + 1), 192, 192, 6);
        goto LABEL_10;
      }
      v10[1] = v42 + 192;
      if ( ((v39 + 63) & 0xFFFFFFFFFFFFFFC0LL) != 0 )
      {
        v11 = (v39 + 63) & 0xFFFFFFFFFFFFFFC0LL;
        goto LABEL_10;
      }
    }
    v4 = v40;
    v28 = v5 + 20;
    do
    {
      v29 = *(v28 - 3);
      v28 -= 3;
      if ( v29 )
        j_j___libc_free_0_0(v29);
    }
    while ( v28 != v5 + 8 );
    *((_DWORD *)v5 + 40) = 1;
    memset(v5, 0, 0x98u);
    v30 = (__int64 *)((unsigned __int64)dest & 0xFFFFFFFFFFFFFFC0LL);
    v5[10] = *(_QWORD *)(((unsigned __int64)dest & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v53[0] - 1) + 8);
    v5[1] = dest;
    v5[11] = *(_QWORD *)((v55 & 0xFFFFFFFFFFFFFFC0LL) + 16LL * (unsigned int)(v53[1] - 1) + 8);
    v5[2] = v55;
    v31 = *v30;
    *((_DWORD *)v5 + 41) = 2;
    *v5 = v31;
    sub_F038C0((unsigned int *)(v40 + 8), (__int64)(v5 + 1), 2, v41, v14, v9);
    sub_2DF52D0((__int64)&dest, a4);
  }
  sub_2DFB620((unsigned int *)v4, a2, a3, (__int64)&dest);
  if ( dest )
    j_j___libc_free_0_0((unsigned __int64)dest);
}

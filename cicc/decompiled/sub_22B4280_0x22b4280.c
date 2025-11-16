// Function: sub_22B4280
// Address: 0x22b4280
//
__int64 __fastcall sub_22B4280(__int64 a1)
{
  char v1; // r15
  char v2; // r14
  char v3; // bl
  char v4; // r13
  _QWORD *v5; // rax
  char v6; // bl
  _QWORD *v7; // r8
  _QWORD *v8; // rax
  __int64 v9; // rax
  unsigned __int64 v10; // r13
  __int64 v11; // rsi
  __int64 *v12; // r14
  __int64 *v13; // rbx
  __int64 i; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 v17; // rsi
  __int64 *v18; // rbx
  unsigned __int64 v19; // r12
  __int64 v20; // rsi
  __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 *v23; // r14
  __int64 *v24; // rbx
  __int64 j; // rax
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rsi
  __int64 *v29; // rbx
  unsigned __int64 v30; // r12
  __int64 v31; // rsi
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 *v35; // r15
  unsigned __int64 *v36; // r14
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // r12
  __int64 v39; // rsi
  __int64 v40; // rdi
  _QWORD *v41; // [rsp+8h] [rbp-38h]

  v1 = byte_4FDBAC8;
  v2 = LOBYTE(qword_4FDBC48[8]) ^ 1;
  v3 = unk_4FDB9E8;
  v4 = LOBYTE(qword_4FDBB68[8]) ^ 1;
  v5 = (_QWORD *)sub_22077B0(0x158u);
  v6 = v3 ^ 1;
  v7 = v5;
  if ( v5 )
  {
    *v5 = 0;
    v8 = v5 + 4;
    *(v8 - 3) = 0;
    v7[2] = v8;
    v7[3] = 0x400000000LL;
    v7[15] = 0x400000000LL;
    v7[8] = v7 + 10;
    v7[20] = v7 + 22;
    v7[14] = v7 + 16;
    v7[9] = 0;
    v7[10] = 0;
    v7[11] = 0;
    v7[12] = 0;
    v7[13] = 0;
    v7[29] = 0;
    v7[30] = 0;
    v7[31] = 0;
    *((_DWORD *)v7 + 64) = 0;
    *((_DWORD *)v7 + 66) = 0;
    v7[34] = v7;
    v7[35] = v7 + 12;
    v7[36] = 0;
    *((_DWORD *)v7 + 74) = 0;
    v7[21] = 0;
    v7[22] = 16;
    v7[23] = 0;
    v7[24] = 4294967293LL;
    v7[25] = 0;
    v7[26] = 0;
    v7[27] = 0;
    *((_DWORD *)v7 + 56) = 0;
    v41 = v7;
    v9 = sub_9D1E70((__int64)(v7 + 12), 16, 16, 3);
    v7 = v41;
    *(_QWORD *)(v9 + 8) = v9;
    *(_QWORD *)v9 = v9 | 4;
    v41[36] = v9;
    *((_BYTE *)v41 + 304) = v2;
    *((_BYTE *)v41 + 305) = v4;
    *((_BYTE *)v41 + 306) = v1;
    *((_BYTE *)v41 + 307) = v6;
    *((_BYTE *)v41 + 308) = 0;
    *((_BYTE *)v41 + 336) = 0;
  }
  v10 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(a1 + 176) = v7;
  if ( v10 )
  {
    if ( *(_BYTE *)(v10 + 336) )
    {
      v35 = *(unsigned __int64 **)(v10 + 320);
      v36 = *(unsigned __int64 **)(v10 + 312);
      *(_BYTE *)(v10 + 336) = 0;
      if ( v35 != v36 )
      {
        do
        {
          v37 = v36[1];
          v38 = *v36;
          if ( v37 != *v36 )
          {
            do
            {
              v39 = *(unsigned int *)(v38 + 144);
              v40 = *(_QWORD *)(v38 + 128);
              v38 += 152LL;
              sub_C7D6A0(v40, 8 * v39, 4);
              sub_C7D6A0(*(_QWORD *)(v38 - 56), 8LL * *(unsigned int *)(v38 - 40), 4);
              sub_C7D6A0(*(_QWORD *)(v38 - 88), 16LL * *(unsigned int *)(v38 - 72), 8);
              sub_C7D6A0(*(_QWORD *)(v38 - 120), 16LL * *(unsigned int *)(v38 - 104), 8);
            }
            while ( v37 != v38 );
            v38 = *v36;
          }
          if ( v38 )
            j_j___libc_free_0(v38);
          v36 += 3;
        }
        while ( v35 != v36 );
        v36 = *(unsigned __int64 **)(v10 + 312);
      }
      if ( v36 )
        j_j___libc_free_0((unsigned __int64)v36);
    }
    sub_C7D6A0(*(_QWORD *)(v10 + 240), 16LL * *(unsigned int *)(v10 + 256), 8);
    v11 = 16LL * *(unsigned int *)(v10 + 224);
    if ( !*(_DWORD *)(v10 + 224) )
      v11 = 0;
    sub_C7D6A0(*(_QWORD *)(v10 + 208), v11, 8);
    sub_E66D20(v10 + 96);
    v12 = *(__int64 **)(v10 + 112);
    v13 = &v12[*(unsigned int *)(v10 + 120)];
    if ( v12 != v13 )
    {
      for ( i = *(_QWORD *)(v10 + 112); ; i = *(_QWORD *)(v10 + 112) )
      {
        v15 = *v12;
        v16 = (unsigned int)(((__int64)v12 - i) >> 3) >> 7;
        v17 = 4096LL << v16;
        if ( v16 >= 0x1E )
          v17 = 0x40000000000LL;
        ++v12;
        sub_C7D6A0(v15, v17, 16);
        if ( v13 == v12 )
          break;
      }
    }
    v18 = *(__int64 **)(v10 + 160);
    v19 = (unsigned __int64)&v18[2 * *(unsigned int *)(v10 + 168)];
    if ( v18 != (__int64 *)v19 )
    {
      do
      {
        v20 = v18[1];
        v21 = *v18;
        v18 += 2;
        sub_C7D6A0(v21, v20, 16);
      }
      while ( (__int64 *)v19 != v18 );
      v19 = *(_QWORD *)(v10 + 160);
    }
    if ( v19 != v10 + 176 )
      _libc_free(v19);
    v22 = *(_QWORD *)(v10 + 112);
    if ( v22 != v10 + 128 )
      _libc_free(v22);
    sub_22B0CF0(v10);
    v23 = *(__int64 **)(v10 + 16);
    v24 = &v23[*(unsigned int *)(v10 + 24)];
    if ( v23 != v24 )
    {
      for ( j = *(_QWORD *)(v10 + 16); ; j = *(_QWORD *)(v10 + 16) )
      {
        v26 = *v23;
        v27 = (unsigned int)(((__int64)v23 - j) >> 3) >> 7;
        v28 = 4096LL << v27;
        if ( v27 >= 0x1E )
          v28 = 0x40000000000LL;
        ++v23;
        sub_C7D6A0(v26, v28, 16);
        if ( v24 == v23 )
          break;
      }
    }
    v29 = *(__int64 **)(v10 + 64);
    v30 = (unsigned __int64)&v29[2 * *(unsigned int *)(v10 + 72)];
    if ( v29 != (__int64 *)v30 )
    {
      do
      {
        v31 = v29[1];
        v32 = *v29;
        v29 += 2;
        sub_C7D6A0(v32, v31, 16);
      }
      while ( (__int64 *)v30 != v29 );
      v30 = *(_QWORD *)(v10 + 64);
    }
    if ( v30 != v10 + 80 )
      _libc_free(v30);
    v33 = *(_QWORD *)(v10 + 16);
    if ( v33 != v10 + 32 )
      _libc_free(v33);
    j_j___libc_free_0(v10);
  }
  return 0;
}

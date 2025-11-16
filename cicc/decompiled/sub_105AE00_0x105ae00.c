// Function: sub_105AE00
// Address: 0x105ae00
//
_QWORD *__fastcall sub_105AE00(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r8
  _QWORD *v10; // r12
  int v12; // edx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  unsigned int v16; // esi
  __int64 v17; // rcx
  int v18; // r11d
  __int64 *v19; // r8
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r10
  __int64 v23; // r14
  __int64 v24; // rsi
  __int64 v25; // r14
  __int64 v26; // rsi
  _QWORD *v27; // rbx
  _QWORD *v28; // rdi
  int v29; // r9d
  int v30; // eax
  int v31; // edx
  int v32; // eax
  int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // eax
  __int64 v36; // rcx
  int v37; // r10d
  __int64 *v38; // r9
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rdi
  int v42; // r9d
  unsigned int v43; // r15d
  __int64 *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // [rsp+8h] [rbp-98h] BYREF
  __int64 v47[5]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v48[4]; // [rsp+38h] [rbp-68h] BYREF
  __int64 v49; // [rsp+58h] [rbp-48h]
  __int64 v50; // [rsp+60h] [rbp-40h]

  v2 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v2 != a2 + 48 )
  {
    if ( !v2 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 <= 0xA && (unsigned int)sub_B46E30(v2 - 24) > 1 )
    {
      v5 = *(unsigned int *)(a1 + 432);
      v6 = *(_QWORD *)(a1 + 416);
      if ( (_DWORD)v5 )
      {
        v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
        {
LABEL_7:
          if ( v8 != (__int64 *)(v6 + 16 * v5) )
            return (_QWORD *)v8[1];
        }
        else
        {
          v12 = 1;
          while ( v9 != -4096 )
          {
            v29 = v12 + 1;
            v7 = (v5 - 1) & (v12 + v7);
            v8 = (__int64 *)(v6 + 16LL * v7);
            v9 = *v8;
            if ( *v8 == a2 )
              goto LABEL_7;
            v12 = v29;
          }
        }
      }
      v13 = *(_QWORD *)(a1 + 400);
      v47[3] = a2;
      v14 = *(_QWORD *)(a1 + 392);
      v47[0] = a1;
      v47[2] = v13;
      v47[1] = v14;
      v47[4] = v13;
      v48[1] = v48;
      v48[0] = v48;
      v48[2] = 0;
      v48[3] = v48;
      v15 = sub_22077B0(160);
      if ( v15 )
      {
        *(_QWORD *)v15 = 0;
        *(_QWORD *)(v15 + 8) = v15 + 32;
        *(_QWORD *)(v15 + 16) = 4;
        *(_DWORD *)(v15 + 24) = 0;
        *(_BYTE *)(v15 + 28) = 1;
        *(_QWORD *)(v15 + 64) = 0;
        *(_QWORD *)(v15 + 72) = v15 + 96;
        *(_QWORD *)(v15 + 80) = 4;
        *(_DWORD *)(v15 + 88) = 0;
        *(_BYTE *)(v15 + 92) = 1;
        *(_QWORD *)(v15 + 128) = 0;
        *(_QWORD *)(v15 + 136) = 0;
        *(_QWORD *)(v15 + 144) = 0;
        *(_DWORD *)(v15 + 152) = 0;
      }
      v49 = v15;
      v50 = v15 + 128;
      sub_1059B30(&v46, v47);
      v16 = *(_DWORD *)(a1 + 432);
      if ( v16 )
      {
        v17 = *(_QWORD *)(a1 + 416);
        v18 = 1;
        v19 = 0;
        v20 = (v16 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( *v21 == a2 )
        {
LABEL_20:
          v23 = v46;
          v10 = (_QWORD *)v21[1];
          if ( v46 )
          {
            v24 = 16LL * *(unsigned int *)(v46 + 152);
            sub_C7D6A0(*(_QWORD *)(v46 + 136), v24, 8);
            if ( !*(_BYTE *)(v23 + 92) )
              _libc_free(*(_QWORD *)(v23 + 72), v24);
            if ( !*(_BYTE *)(v23 + 28) )
              _libc_free(*(_QWORD *)(v23 + 8), v24);
            j_j___libc_free_0(v23, 160);
          }
LABEL_26:
          v25 = v49;
          if ( v49 )
          {
            v26 = 16LL * *(unsigned int *)(v49 + 152);
            sub_C7D6A0(*(_QWORD *)(v49 + 136), v26, 8);
            if ( !*(_BYTE *)(v25 + 92) )
              _libc_free(*(_QWORD *)(v25 + 72), v26);
            if ( !*(_BYTE *)(v25 + 28) )
              _libc_free(*(_QWORD *)(v25 + 8), v26);
            j_j___libc_free_0(v25, 160);
          }
          v27 = (_QWORD *)v48[0];
          while ( v27 != v48 )
          {
            v28 = v27;
            v27 = (_QWORD *)*v27;
            j_j___libc_free_0(v28, 40);
          }
          return v10;
        }
        while ( v22 != -4096 )
        {
          if ( !v19 && v22 == -8192 )
            v19 = v21;
          v20 = (v16 - 1) & (v18 + v20);
          v21 = (__int64 *)(v17 + 16LL * v20);
          v22 = *v21;
          if ( *v21 == a2 )
            goto LABEL_20;
          ++v18;
        }
        if ( !v19 )
          v19 = v21;
        v30 = *(_DWORD *)(a1 + 424);
        ++*(_QWORD *)(a1 + 408);
        v31 = v30 + 1;
        if ( 4 * (v30 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 428) - v31 > v16 >> 3 )
          {
LABEL_47:
            *(_DWORD *)(a1 + 424) = v31;
            if ( *v19 != -4096 )
              --*(_DWORD *)(a1 + 428);
            *v19 = a2;
            v10 = (_QWORD *)v46;
            v19[1] = v46;
            goto LABEL_26;
          }
          sub_1058DA0(a1 + 408, v16);
          v39 = *(_DWORD *)(a1 + 432);
          if ( v39 )
          {
            v40 = v39 - 1;
            v41 = *(_QWORD *)(a1 + 416);
            v42 = 1;
            v43 = (v39 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
            v31 = *(_DWORD *)(a1 + 424) + 1;
            v44 = 0;
            v19 = (__int64 *)(v41 + 16LL * v43);
            v45 = *v19;
            if ( *v19 != a2 )
            {
              while ( v45 != -4096 )
              {
                if ( v45 == -8192 && !v44 )
                  v44 = v19;
                v43 = v40 & (v42 + v43);
                v19 = (__int64 *)(v41 + 16LL * v43);
                v45 = *v19;
                if ( *v19 == a2 )
                  goto LABEL_47;
                ++v42;
              }
              if ( v44 )
                v19 = v44;
            }
            goto LABEL_47;
          }
LABEL_74:
          ++*(_DWORD *)(a1 + 424);
          BUG();
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 408);
      }
      sub_1058DA0(a1 + 408, 2 * v16);
      v32 = *(_DWORD *)(a1 + 432);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = *(_QWORD *)(a1 + 416);
        v35 = (v32 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v31 = *(_DWORD *)(a1 + 424) + 1;
        v19 = (__int64 *)(v34 + 16LL * v35);
        v36 = *v19;
        if ( *v19 != a2 )
        {
          v37 = 1;
          v38 = 0;
          while ( v36 != -4096 )
          {
            if ( v36 == -8192 && !v38 )
              v38 = v19;
            v35 = v33 & (v37 + v35);
            v19 = (__int64 *)(v34 + 16LL * v35);
            v36 = *v19;
            if ( *v19 == a2 )
              goto LABEL_47;
            ++v37;
          }
          if ( v38 )
            v19 = v38;
        }
        goto LABEL_47;
      }
      goto LABEL_74;
    }
  }
  v10 = qword_4F8FBE0;
  if ( !byte_4F8FBD8[0] && (unsigned int)sub_2207590(byte_4F8FBD8) )
  {
    BYTE4(qword_4F8FBE0[3]) = 1;
    qword_4F8FBE0[1] = &qword_4F8FBE0[4];
    qword_4F8FBE0[0] = 0;
    qword_4F8FBE0[2] = 4;
    LODWORD(qword_4F8FBE0[3]) = 0;
    qword_4F8FBE0[8] = 0;
    qword_4F8FBE0[9] = &qword_4F8FBE0[12];
    qword_4F8FBE0[10] = 4;
    LODWORD(qword_4F8FBE0[11]) = 0;
    BYTE4(qword_4F8FBE0[11]) = 1;
    qword_4F8FBE0[16] = 0;
    qword_4F8FBE0[17] = 0;
    qword_4F8FBE0[18] = 0;
    LODWORD(qword_4F8FBE0[19]) = 0;
    __cxa_atexit((void (*)(void *))sub_1055C50, qword_4F8FBE0, &qword_4A427C0);
    sub_2207640(byte_4F8FBD8);
  }
  return v10;
}

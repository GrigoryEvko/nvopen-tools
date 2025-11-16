// Function: sub_2FB8410
// Address: 0x2fb8410
//
void __fastcall sub_2FB8410(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r15
  _QWORD *v11; // r10
  _QWORD *v12; // rbx
  _QWORD *v13; // r9
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // rax
  __int16 v18; // ax
  __int64 v19; // rcx
  unsigned __int64 v20; // r8
  _QWORD *v21; // r9
  __int64 v22; // rsi
  __int64 v23; // r10
  __int64 v24; // rdx
  __int64 v25; // r11
  __int64 *v26; // rcx
  unsigned int v27; // esi
  __int64 v28; // rax
  __int64 v29; // r11
  unsigned __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // edx
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rcx
  __int64 v36; // rax
  unsigned __int64 v37; // rsi
  __int64 v38; // rcx
  unsigned __int64 i; // rax
  __int64 j; // rdi
  __int16 v41; // dx
  unsigned int v42; // esi
  __int64 v43; // r10
  unsigned int v44; // ecx
  __int64 *v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // rbx
  char v48; // al
  unsigned __int64 v49; // rbx
  __int64 v50; // rcx
  __int64 v51; // rbx
  __int64 *v52; // rax
  int *v53; // r9
  int v54; // edx
  int v55; // r9d
  __int64 v56; // [rsp+0h] [rbp-D0h]
  __int64 v57; // [rsp+8h] [rbp-C8h]
  __int64 v58; // [rsp+18h] [rbp-B8h]
  __int64 v59; // [rsp+20h] [rbp-B0h]
  __int64 v60; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v61; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v62; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v63; // [rsp+30h] [rbp-A0h]
  unsigned int v64; // [rsp+30h] [rbp-A0h]
  __int64 v65; // [rsp+38h] [rbp-98h]
  __int64 v66; // [rsp+38h] [rbp-98h]
  __int64 v67; // [rsp+40h] [rbp-90h] BYREF
  _BYTE *v68; // [rsp+48h] [rbp-88h] BYREF
  __int64 v69; // [rsp+50h] [rbp-80h]
  _BYTE v70[120]; // [rsp+58h] [rbp-78h] BYREF

  v7 = sub_2DF8570(
         *(_QWORD *)(a1 + 8),
         *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL) + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL)),
         *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL),
         *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL),
         a5,
         a6);
  v8 = *((unsigned int *)a2 + 2);
  v58 = v7;
  v68 = v70;
  v69 = 0x400000000LL;
  v67 = a1 + 192;
  v59 = *a2 + 8 * v8;
  if ( *a2 != v59 )
  {
    v9 = *a2;
    while ( 1 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)v9 + 8LL);
      v11 = *(_QWORD **)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      v12 = v11;
      v13 = *(_QWORD **)(v11[3] + 56LL);
      if ( v11 == v13 )
      {
LABEL_13:
        v16 = (unsigned __int64)v12;
      }
      else
      {
        while ( 1 )
        {
          v14 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v14 )
            BUG();
          v15 = *(_QWORD *)v14;
          v16 = *v12 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 4) != 0 )
          {
            while ( 1 )
            {
              v17 = v15 & 0xFFFFFFFFFFFFFFF8LL;
              v16 = v17;
              if ( (*(_BYTE *)(v17 + 44) & 4) == 0 )
                break;
              v15 = *(_QWORD *)v17;
            }
          }
          v18 = *(_WORD *)(v16 + 68);
          if ( (unsigned __int16)(v18 - 14) > 4u && v18 != 24 )
            break;
          v12 = (_QWORD *)v16;
          if ( (_QWORD *)v16 == v13 )
            goto LABEL_13;
        }
      }
      v60 = *(_QWORD *)(v11[3] + 56LL);
      v63 = *(_QWORD *)(*(_QWORD *)v9 + 8LL) & 0xFFFFFFFFFFFFFFF8LL;
      v65 = *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 16);
      sub_2E14FC0(*(_QWORD *)(a1 + 8), v58, *(_QWORD *)(*(_QWORD *)v9 + 8LL));
      sub_2FAD510(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL), v65);
      sub_2E88E20(v65);
      v20 = v63;
      v21 = (_QWORD *)v60;
      v66 = (v10 >> 1) & 3;
      if ( ((v10 >> 1) & 3) != 0 )
        v22 = v63 | (2LL * ((int)v66 - 1));
      else
        v22 = *(_QWORD *)v63 & 0xFFFFFFFFFFFFFFF8LL | 6;
      v23 = v67;
      v24 = *(unsigned int *)(v67 + 184);
      if ( (_DWORD)v24 )
      {
        sub_2FB5930((__int64)&v67, v22, v24, v19, v63, v60);
        v31 = (unsigned int)v69;
        v20 = v63;
        v21 = (_QWORD *)v60;
      }
      else
      {
        v25 = *(unsigned int *)(v67 + 188);
        if ( (_DWORD)v25 )
        {
          v26 = (__int64 *)(v67 + 8);
          v27 = *(_DWORD *)((v22 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (v22 >> 1) & 3;
          do
          {
            if ( (*(_DWORD *)((*v26 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v26 >> 1) & 3) > v27 )
              break;
            v24 = (unsigned int)(v24 + 1);
            v26 += 2;
          }
          while ( (_DWORD)v25 != (_DWORD)v24 );
        }
        LODWORD(v69) = 0;
        v28 = 0;
        v29 = (v24 << 32) | v25;
        if ( !HIDWORD(v69) )
        {
          v56 = v29;
          v57 = v67;
          sub_C8D5F0((__int64)&v68, v70, 1u, 0x10u, v63, v60);
          v29 = v56;
          v23 = v57;
          v21 = (_QWORD *)v60;
          v20 = v63;
          v28 = 16LL * (unsigned int)v69;
        }
        v30 = (unsigned __int64)v68;
        *(_QWORD *)&v68[v28] = v23;
        *(_QWORD *)(v30 + v28 + 8) = v29;
        v31 = (unsigned int)(v69 + 1);
        LODWORD(v69) = v69 + 1;
      }
      if ( !(_DWORD)v31 || *((_DWORD *)v68 + 3) >= *((_DWORD *)v68 + 2) )
        goto LABEL_46;
      v32 = *(_DWORD *)(v20 + 24);
      v33 = (unsigned __int64)&v68[16 * v31 - 16];
      v34 = *(unsigned int *)(v33 + 12);
      v35 = *(_QWORD *)v33;
      v36 = *(_QWORD *)(*(_QWORD *)v33 + 16 * v34);
      if ( *(_DWORD *)(v67 + 184) )
      {
        if ( ((unsigned int)v66 | v32) <= (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3) )
          goto LABEL_46;
      }
      else if ( (*(_DWORD *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v36 >> 1) & 3) >= ((unsigned int)v66
                                                                                                  | v32) )
      {
        goto LABEL_46;
      }
      if ( v10 != *(_QWORD *)(v35 + 16 * v34 + 8) )
        goto LABEL_46;
      v64 = *(_DWORD *)(v35 + 4 * v34 + 144);
      if ( v12 == v21 )
        goto LABEL_42;
      v37 = v16;
      v38 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
      for ( i = v16; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        ;
      if ( (*(_DWORD *)(v16 + 44) & 8) != 0 )
      {
        do
          v37 = *(_QWORD *)(v37 + 8);
        while ( (*(_BYTE *)(v37 + 44) & 8) != 0 );
      }
      for ( j = *(_QWORD *)(v37 + 8); j != i; i = *(_QWORD *)(i + 8) )
      {
        v41 = *(_WORD *)(i + 68);
        if ( (unsigned __int16)(v41 - 14) > 4u && v41 != 24 )
          break;
      }
      v42 = *(_DWORD *)(v38 + 144);
      v43 = *(_QWORD *)(v38 + 128);
      if ( v42 )
      {
        v44 = (v42 - 1) & (((unsigned int)i >> 4) ^ ((unsigned int)i >> 9));
        v45 = (__int64 *)(v43 + 16LL * v44);
        v46 = *v45;
        if ( i == *v45 )
          goto LABEL_40;
        v54 = 1;
        while ( v46 != -4096 )
        {
          v55 = v54 + 1;
          v44 = (v42 - 1) & (v54 + v44);
          v45 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v45;
          if ( i == *v45 )
            goto LABEL_40;
          v54 = v55;
        }
      }
      v45 = (__int64 *)(v43 + 16LL * v42);
LABEL_40:
      v47 = v45[1];
      v61 = v20;
      v48 = sub_2E89D80(v16, *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL) + 112LL), 0);
      v20 = v61;
      if ( !v48
        || (v49 = v47 & 0xFFFFFFFFFFFFFFF8LL,
            v50 = *(_QWORD *)(*(_QWORD *)&v68[16 * (unsigned int)v69 - 16]
                            + 16LL * *(unsigned int *)&v68[16 * (unsigned int)v69 - 4]),
            (*(_DWORD *)(v49 + 24) | 2u) <= (*(_DWORD *)((v50 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                           | (unsigned int)(v50 >> 1) & 3)) )
      {
LABEL_42:
        v62 = v20;
        v51 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
        v52 = (__int64 *)sub_2E09D00((__int64 *)v51, v10);
        v53 = 0;
        if ( v52 != (__int64 *)(*(_QWORD *)v51 + 24LL * *(unsigned int *)(v51 + 8))
          && (*(_DWORD *)((*v52 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v52 >> 1) & 3)) <= (*(_DWORD *)(v62 + 24) | (unsigned int)v66) )
        {
          v53 = (int *)v52[2];
        }
        sub_2FB7E60(a1, v64, v53);
        goto LABEL_46;
      }
      sub_2FB7090((__int64)&v67, v49 | 4);
LABEL_46:
      v9 += 8;
      if ( v59 == v9 )
      {
        if ( v68 != v70 )
          _libc_free((unsigned __int64)v68);
        return;
      }
    }
  }
}

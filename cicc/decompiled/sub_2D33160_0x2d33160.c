// Function: sub_2D33160
// Address: 0x2d33160
//
void __fastcall sub_2D33160(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // r14
  __int64 *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r8
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r13
  unsigned __int64 v13; // rbx
  char v14; // r12
  unsigned int v15; // r15d
  __int64 v16; // r10
  __int64 v17; // rax
  char v18; // r15
  unsigned __int64 v19; // r8
  char v20; // al
  unsigned int v21; // r10d
  unsigned int v22; // esi
  __int64 v23; // r11
  int v24; // ecx
  unsigned __int64 v25; // rdi
  __int64 v26; // rcx
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned int v31; // eax
  unsigned int v32; // edx
  unsigned int v33; // edi
  char v34; // dl
  __int64 *v35; // r10
  __int64 v36; // rbx
  __int64 v37; // rsi
  unsigned int v38; // ecx
  unsigned int i; // edx
  unsigned __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  unsigned __int64 v46; // [rsp+0h] [rbp-140h]
  __int64 v47; // [rsp+8h] [rbp-138h]
  __int64 v48; // [rsp+10h] [rbp-130h]
  __int64 v49; // [rsp+20h] [rbp-120h]
  char v50; // [rsp+2Eh] [rbp-112h]
  char v51; // [rsp+2Fh] [rbp-111h]
  __int64 v52; // [rsp+30h] [rbp-110h]
  unsigned __int64 v54; // [rsp+48h] [rbp-F8h]
  __int64 v55; // [rsp+48h] [rbp-F8h]
  __int64 v56[2]; // [rsp+50h] [rbp-F0h] BYREF
  char v57[16]; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v58; // [rsp+70h] [rbp-D0h]
  char v59; // [rsp+80h] [rbp-C0h]
  char *v60; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v61; // [rsp+98h] [rbp-A8h]
  char v62; // [rsp+A0h] [rbp-A0h] BYREF
  _BYTE *v63; // [rsp+C0h] [rbp-80h] BYREF
  unsigned __int64 v64; // [rsp+C8h] [rbp-78h]
  _BYTE v65[48]; // [rsp+D0h] [rbp-70h] BYREF
  int v66; // [rsp+100h] [rbp-40h]

  v2 = *a1;
  v63 = (_BYTE *)(a2 & 0xFFFFFFFFFFFFFFFBLL);
  v46 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v3 = sub_2D2B870((_QWORD *)(v2 + 72), (__int64 *)&v63);
  if ( !v3 )
    return;
  v50 = 0;
  v61 = 0x100000000LL;
  v4 = *(_QWORD *)(v3 + 16);
  v5 = *(unsigned int *)(v3 + 24);
  v60 = &v62;
  v49 = v4;
  v6 = v4 + 32 * v5;
  if ( v4 == v6 )
    goto LABEL_37;
  v7 = v6 - 32;
  do
  {
    v54 = v7;
    v8 = (__int64 *)(*(_QWORD *)(*a1 + 48) + 40LL * (unsigned int)(*(_DWORD *)v7 - 1));
    v9 = *v8;
    v56[1] = v8[4];
    v56[0] = v9;
    v63 = (_BYTE *)sub_AF3FE0(v9);
    v64 = v11;
    v51 = v11;
    if ( !(_BYTE)v11 )
      goto LABEL_33;
    v12 = (unsigned __int64)v63;
    v13 = (v63 != 0) + ((unsigned __int64)&v63[-(v63 != 0)] >> 3);
    if ( v13 - 1 > 0x7FF )
      goto LABEL_33;
    v14 = (v63 != 0) + ((unsigned __int64)&v63[-(v63 != 0)] >> 3);
    v15 = (unsigned int)(v13 + 63) >> 6;
    v63 = v65;
    v16 = a1[1];
    v64 = 0x600000000LL;
    v17 = 8LL * v15;
    if ( v15 > 6 )
    {
      v47 = 8LL * v15;
      v48 = v16;
      sub_C8D5F0((__int64)&v63, v65, v15, 8u, v10, (__int64)v65);
      v45 = (__int64)v63;
      *(_QWORD *)v63 = 0;
      *(_QWORD *)(v45 + v47 - 8) = 0;
      memset(
        (void *)((v45 + 8) & 0xFFFFFFFFFFFFFFF8LL),
        0,
        8LL * (((unsigned int)v47 + (_DWORD)v45 - (((_DWORD)v45 + 8) & 0xFFFFFFF8)) >> 3));
      LODWORD(v64) = (unsigned int)(v13 + 63) >> 6;
      v16 = v48;
    }
    else
    {
      if ( v17 )
      {
        *(_QWORD *)&v65[8 * v15 - 8] = 0;
        memset(v65, 0, 8LL * ((unsigned int)(v17 - 1) >> 3));
      }
      LODWORD(v64) = (unsigned int)(v13 + 63) >> 6;
    }
    v66 = v13;
    sub_2D32F70((__int64)v57, v16, v56, (__int64)&v63);
    if ( v63 != v65 )
      _libc_free((unsigned __int64)v63);
    v18 = v59;
    v52 = v58;
    sub_AF47B0(
      (__int64)&v63,
      *(unsigned __int64 **)(*(_QWORD *)(v7 + 8) + 16LL),
      *(unsigned __int64 **)(*(_QWORD *)(v7 + 8) + 24LL));
    if ( !v65[0] )
    {
      if ( v18 )
      {
        v23 = *(_QWORD *)(v52 + 16);
        LODWORD(v19) = 0;
LABEL_27:
        v31 = v13;
        v32 = v19;
LABEL_28:
        v33 = v32;
        v34 = v32 & 0x3F;
        v33 >>= 6;
        v35 = (__int64 *)(v23 + 8LL * v33);
        v36 = *v35;
        v37 = 1LL << v31;
        if ( v33 == v31 >> 6 )
        {
          *v35 = (v37 - (1LL << v34)) | v36;
        }
        else
        {
          *v35 = (-1LL << v34) | v36;
          v38 = (((_DWORD)v19 != 0) + (unsigned int)(((unsigned int)v19 - (unsigned __int64)((_DWORD)v19 != 0)) >> 6)) << 6;
          for ( i = v38 + 64; i <= v31; i += 64 )
          {
            *(_QWORD *)(*(_QWORD *)(v52 + 16) + 8LL * ((i - 64) >> 6)) = -1;
            v38 = i;
          }
          if ( v38 < v31 )
            *(_QWORD *)(*(_QWORD *)(v52 + 16) + 8LL * (v38 >> 6)) |= v37 - 1;
        }
LABEL_33:
        sub_2D29B40((unsigned int *)&v60, v7);
        goto LABEL_34;
      }
      v20 = 0;
      v21 = 0;
      LODWORD(v19) = 0;
      v22 = (unsigned int)(v13 - 1) >> 6;
LABEL_17:
      v23 = *(_QWORD *)(v52 + 16);
      v24 = 64 - (v20 & 0x3F);
      v25 = 0xFFFFFFFFFFFFFFFFLL >> v24;
      if ( v24 == 64 )
        v25 = 0;
      v26 = v21;
      while ( 1 )
      {
        v27 = *(_QWORD *)(v23 + 8 * v26);
        v28 = ~v27;
        _RAX = ~(v25 | v27);
        if ( v21 != (_DWORD)v26 )
          _RAX = v28;
        if ( v22 == (_DWORD)v26 )
          _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -v14;
        if ( _RAX )
          break;
        if ( v22 < (unsigned int)++v26 )
          goto LABEL_41;
      }
      __asm { tzcnt   rax, rax }
      if ( ((_DWORD)v26 << 6) + (_DWORD)_RAX != -1 )
        goto LABEL_27;
      goto LABEL_41;
    }
    v19 = v64 >> 3;
    v13 = (&v63[v64] != 0) + ((unsigned __int64)&v63[v64 - (&v63[v64] != 0)] >> 3);
    if ( v12 < (unsigned __int64)&v63[v64] )
      goto LABEL_33;
    if ( v18 )
    {
      if ( v12 >= (unsigned __int64)&v63[v64] )
      {
        v31 = (&v63[v64] != 0) + (unsigned int)((unsigned __int64)&v63[v64 - (&v63[v64] != 0)] >> 3);
        v32 = v64 >> 3;
        if ( (_DWORD)v13 != (_DWORD)v19 )
        {
          v23 = *(_QWORD *)(v52 + 16);
          goto LABEL_28;
        }
      }
      goto LABEL_33;
    }
    v14 = (&v63[v64] != 0) + ((unsigned __int64)&v63[v64 - (&v63[v64] != 0)] >> 3);
    v20 = v64 >> 3;
    if ( (_DWORD)v13 != (_DWORD)v19 )
    {
      v21 = (unsigned int)v19 >> 6;
      v22 = (unsigned int)(v13 - 1) >> 6;
      if ( (unsigned int)v19 >> 6 <= v22 )
        goto LABEL_17;
    }
LABEL_41:
    v50 = v51;
LABEL_34:
    v7 -= 32LL;
  }
  while ( v49 != v54 );
  if ( v50 )
  {
    sub_2D24B00((unsigned __int64)v60, (__int64)&v60[32 * (unsigned int)v61]);
    v55 = *a1;
    v63 = (_BYTE *)v46;
    v40 = sub_2D2B8B0((unsigned __int64 *)(v55 + 72), (__int64 *)&v63);
    sub_2D29780((unsigned int *)v40, (__int64)&v60, v41, v42, v43, v44);
    *(_BYTE *)a1[2] = 1;
  }
LABEL_37:
  sub_2D288B0((__int64)&v60);
}

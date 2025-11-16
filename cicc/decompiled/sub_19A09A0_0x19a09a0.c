// Function: sub_19A09A0
// Address: 0x19a09a0
//
__int64 __fastcall sub_19A09A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6)
{
  __int64 result; // rax
  __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // r12
  __int64 *v13; // r14
  __int64 *v14; // rbx
  unsigned __int64 v15; // rcx
  __int64 v16; // rdx
  _BYTE *v17; // rdi
  unsigned __int64 v18; // r8
  int v19; // eax
  char v20; // al
  __int64 v21; // rcx
  __int16 v22; // ax
  _BYTE *v23; // rsi
  _BYTE *v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rdx
  _BYTE *v30; // rsi
  __int64 v31; // rdx
  __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  char v34; // al
  __int64 v36; // [rsp+8h] [rbp-C8h]
  __int64 v37; // [rsp+10h] [rbp-C0h]
  char v38; // [rsp+1Fh] [rbp-B1h]
  __int64 v39; // [rsp+20h] [rbp-B0h]
  unsigned __int64 v40; // [rsp+28h] [rbp-A8h]
  __int64 v41; // [rsp+30h] [rbp-A0h]
  __int64 v42; // [rsp+38h] [rbp-98h]
  char v43; // [rsp+38h] [rbp-98h]
  __int64 v44; // [rsp+38h] [rbp-98h]
  __int64 v45; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 v46; // [rsp+48h] [rbp-88h]
  char v47; // [rsp+50h] [rbp-80h]
  __int64 v48; // [rsp+58h] [rbp-78h]
  _BYTE *v49; // [rsp+60h] [rbp-70h] BYREF
  __int64 v50; // [rsp+68h] [rbp-68h]
  _BYTE v51[32]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v52; // [rsp+90h] [rbp-40h]
  __int64 v53; // [rsp+98h] [rbp-38h]

  result = *(unsigned int *)(a1 + 376);
  v7 = *(_QWORD *)(a1 + 368);
  v36 = result;
  v8 = v7 + 1984 * result;
  if ( v7 == v8 )
    return result;
  result = *(_QWORD *)(a1 + 368);
  v9 = 1;
  while ( 1 )
  {
    v10 = *(unsigned int *)(result + 752);
    if ( v10 > 0xFFFE )
      break;
    v9 *= v10;
    if ( v9 > 0x3FFFB )
      break;
    result += 1984;
    if ( v8 == result )
    {
      if ( v9 <= 0xFFFE )
        return result;
      break;
    }
  }
  if ( !v36 )
    return result;
  v37 = 0;
  v39 = 0;
  while ( 2 )
  {
    v11 = v37 + v7;
    v40 = *(unsigned int *)(v11 + 752);
    if ( v40 <= 1 )
      goto LABEL_37;
    v41 = 0;
    v38 = 0;
    do
    {
      v12 = *(_QWORD *)(v11 + 744) + 96 * v41;
      v13 = *(__int64 **)(v12 + 32);
      v14 = &v13[*(unsigned int *)(v12 + 40)];
      if ( v13 == v14 )
      {
LABEL_33:
        ++v41;
        continue;
      }
      while ( 1 )
      {
        v21 = *v13;
        v22 = *(_WORD *)(*v13 + 24);
        if ( !v22 )
        {
          v23 = v51;
          v24 = v51;
          v45 = *(_QWORD *)v12;
          v25 = *(_QWORD *)(v12 + 8);
          v46 = v25;
          v47 = *(_BYTE *)(v12 + 16);
          v26 = *(_QWORD *)(v12 + 24);
          v49 = v51;
          v48 = v26;
          v50 = 0x400000000LL;
          v19 = *(_DWORD *)(v12 + 40);
          if ( v19 )
          {
            v42 = v21;
            sub_19930D0((__int64)&v49, v12 + 32, v25, v21, a5, a6);
            v24 = v49;
            v25 = v46;
            v21 = v42;
            v19 = v50;
            v23 = &v49[8 * (unsigned int)v50];
          }
          v52 = *(_QWORD *)(v12 + 80);
          v53 = *(_QWORD *)(v12 + 88);
          v27 = *(_QWORD *)(v21 + 32);
          a6 = *(_DWORD *)(v27 + 32);
          if ( a6 <= 0x40 )
            v15 = (__int64)(*(_QWORD *)(v27 + 24) << (64 - (unsigned __int8)a6)) >> (64 - (unsigned __int8)a6);
          else
            v15 = **(_QWORD **)(v27 + 24);
          v46 = v15 + v25;
          v16 = (__int64)v13 - *(_QWORD *)(v12 + 32);
          v17 = &v24[v16];
          v18 = (unsigned __int64)(v17 + 8);
          if ( v17 + 8 != v23 )
          {
            memmove(v17, v17 + 8, (size_t)&v23[-v18]);
            v19 = v50;
          }
          LODWORD(v50) = v19 - 1;
          goto LABEL_16;
        }
        if ( v22 == 10 )
        {
          v28 = *(_QWORD *)(v21 - 8);
          if ( *(_BYTE *)(v28 + 16) <= 3u && !*(_QWORD *)v12 )
            break;
        }
LABEL_19:
        if ( v14 == ++v13 )
          goto LABEL_33;
      }
      v45 = 0;
      v15 = (unsigned __int64)v51;
      v46 = *(_QWORD *)(v12 + 8);
      v47 = *(_BYTE *)(v12 + 16);
      v29 = *(_QWORD *)(v12 + 24);
      v50 = 0x400000000LL;
      v30 = v51;
      v48 = v29;
      v49 = v51;
      v31 = *(unsigned int *)(v12 + 40);
      if ( (_DWORD)v31 )
      {
        v44 = v28;
        sub_19930D0((__int64)&v49, v12 + 32, v31, (__int64)v51, a5, a6);
        v15 = (unsigned __int64)v49;
        v28 = v44;
        LODWORD(v31) = v50;
        v30 = &v49[8 * (unsigned int)v50];
      }
      v52 = *(_QWORD *)(v12 + 80);
      v32 = *(_QWORD *)(v12 + 88);
      v45 = v28;
      v53 = v32;
      v33 = (unsigned __int64)v13 + v15 - *(_QWORD *)(v12 + 32);
      v18 = v33 + 8;
      if ( v30 != (_BYTE *)(v33 + 8) )
      {
        memmove((void *)v33, (const void *)(v33 + 8), (size_t)&v30[-v18]);
        LODWORD(v31) = v50;
      }
      v16 = (unsigned int)(v31 - 1);
      LODWORD(v50) = v16;
LABEL_16:
      v20 = sub_19A07E0(v11, (__int64)&v45, v16, v15, v18, a6);
      if ( !v20 )
      {
        if ( v49 != v51 )
          _libc_free((unsigned __int64)v49);
        goto LABEL_19;
      }
      v43 = v20;
      sub_1994A60(v11, (__int64 *)v12);
      --v40;
      v34 = v43;
      if ( v49 != v51 )
      {
        _libc_free((unsigned __int64)v49);
        v34 = v43;
      }
      v38 = v34;
    }
    while ( v40 != v41 );
    if ( v38 )
      sub_1996C50(v11, v39, a1 + 32128);
LABEL_37:
    result = ++v39;
    v37 += 1984;
    if ( v39 != v36 )
    {
      v7 = *(_QWORD *)(a1 + 368);
      continue;
    }
    return result;
  }
}

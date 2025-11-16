// Function: sub_777910
// Address: 0x777910
//
__int64 __fastcall sub_777910(__int64 a1, unsigned __int64 a2, unsigned int a3, int a4, FILE *a5, __int64 a6, int *a7)
{
  __int64 v7; // r10
  unsigned __int64 v9; // r12
  char v10; // al
  unsigned int v11; // r13d
  unsigned __int64 v12; // rbx
  unsigned __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // r14
  __int64 v20; // r10
  unsigned int v21; // ecx
  int v22; // edx
  unsigned int v23; // edi
  __int64 v24; // r8
  unsigned int v25; // eax
  int *v26; // rsi
  int v27; // r11d
  int v28; // eax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rax
  char v33; // cl
  __int64 v34; // rax
  char v35; // si
  int *v37; // rsi
  FILE *v38; // rsi
  _QWORD *v39; // rdx
  unsigned int v40; // edi
  unsigned __int64 v41; // rcx
  char v42; // al
  void *v43; // rax
  int v44; // eax
  unsigned int v45; // [rsp+0h] [rbp-70h]
  int v46; // [rsp+4h] [rbp-6Ch]
  __int64 v47; // [rsp+8h] [rbp-68h]
  int v48; // [rsp+8h] [rbp-68h]
  unsigned int v49; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+28h] [rbp-48h]
  int v56[13]; // [rsp+3Ch] [rbp-34h] BYREF

  v7 = a1;
  v9 = a2;
  v10 = *(_BYTE *)(a2 + 140);
  for ( v56[0] = 1; v10 == 12; v10 = *(_BYTE *)(v9 + 140) )
    v9 = *(_QWORD *)(v9 + 160);
  v11 = a3;
  v12 = v9;
  if ( v10 != 8 )
  {
LABEL_4:
    if ( (unsigned __int8)(*(_BYTE *)(v12 + 140) - 2) <= 1u )
    {
      v51 = 16;
      v43 = &loc_1000000;
    }
    else
    {
      v13 = v12;
      v14 = sub_7764B0(a1, v12, v56);
      v51 = v14;
      if ( !v56[0] )
        return 0;
      v7 = a1;
      v16 = 64;
      v17 = 64;
      v46 = 64;
      if ( !v14 )
        goto LABEL_7;
      LODWORD(v43) = 0x10000000 / v14;
    }
    if ( v11 > (unsigned int)v43 )
      goto LABEL_21;
    v44 = v11 * v51;
    if ( (((_BYTE)v11 * (_BYTE)v51) & 7) != 0 )
      v44 = v11 * v51 + 8 - (((_BYTE)v11 * (_BYTE)v51) & 7);
    v15 = (unsigned int)(v44 + 7) >> 3;
    v17 = (unsigned int)(v15 + 57);
    v13 = ((_BYTE)v15 + 57) & 7;
    if ( (((_BYTE)v15 + 57) & 7) != 0 )
      v17 = (unsigned int)(v15 + 65 - v13);
    v16 = (unsigned int)(v44 + v17);
    v46 = v44 + v17;
LABEL_7:
    v47 = v7;
    v49 = v17;
    v18 = j_malloc(v16, v13, v15, v17, a5, a6);
    v7 = v47;
    v19 = v18;
    if ( v18 )
    {
      memset((void *)(v18 + 48), 0, v49 - 48);
      v20 = v47;
      v21 = v49;
      v22 = *(_DWORD *)(v47 + 128) + 1;
      *(_DWORD *)(v47 + 128) = v22;
      *(_DWORD *)(v19 + 32) = v22;
      v23 = *(_DWORD *)(v47 + 64);
      v24 = *(_QWORD *)(v47 + 56);
      v25 = v23 & v22;
      v26 = (int *)(v24 + 4LL * (v23 & v22));
      v27 = *v26;
      *v26 = v22;
      if ( v27 )
      {
        do
        {
          v25 = v23 & (v25 + 1);
          v37 = (int *)(v24 + 4LL * v25);
        }
        while ( *v37 );
        *v37 = v27;
      }
      v28 = *(_DWORD *)(v47 + 68) + 1;
      *(_DWORD *)(v47 + 68) = v28;
      if ( 2 * v28 > v23 )
      {
        v45 = v49;
        v48 = v22;
        v50 = v20;
        sub_7702C0(v20 + 56);
        v21 = v45;
        v22 = v48;
        v20 = v50;
      }
      *(_QWORD *)(v19 + 16) = v12;
      v29 = *(_QWORD *)&a5->_flags;
      *(_DWORD *)(v19 + 40) = v21;
      *(_DWORD *)(v19 + 44) = v11;
      *(_QWORD *)(v19 + 24) = v29;
      *(_DWORD *)(v19 + 36) = v46;
      v30 = *(_QWORD *)(v20 + 184);
      *(_QWORD *)(v19 + 8) = 0;
      *(_QWORD *)v19 = v30;
      v31 = *(_QWORD *)(v20 + 184);
      if ( v31 )
        *(_QWORD *)(v31 + 8) = v19;
      v32 = v21;
      *(_QWORD *)(v20 + 184) = v19;
      v33 = 2;
      v34 = v19 + v32;
      *(_OWORD *)(a6 + 8) = 0;
      *(_QWORD *)a6 = v34;
      *(_QWORD *)(a6 + 24) = v34;
      *(_QWORD *)(a6 + 16) = v34;
      *(_QWORD *)(v34 - 8) = v9;
      if ( a4 )
      {
        v35 = *(_BYTE *)(a6 + 8);
        *(_BYTE *)(a6 + 8) = v35 | 8;
        v33 = 6;
        *(_DWORD *)(a6 + 8) = *(unsigned __int8 *)(a6 + 8) | (a3 << 8);
        if ( !v11 )
          *(_BYTE *)(a6 + 8) = v35 | 0xA;
      }
      *(_DWORD *)(a6 + 12) = v22;
      *a7 = v51;
      *(_BYTE *)(*(_QWORD *)(a6 + 24) - 9LL) |= v33;
      return v19;
    }
LABEL_21:
    if ( (*(_BYTE *)(v7 + 132) & 0x20) == 0 )
    {
      v55 = v7;
      v38 = a5;
      v39 = (_QWORD *)(v7 + 96);
      v40 = 2990;
      goto LABEL_23;
    }
    return 0;
  }
  if ( *(char *)(v9 + 168) >= 0 )
  {
    do
    {
      v41 = *(_QWORD *)(v12 + 176);
      if ( v41 )
      {
        if ( v11 > 0x10000000 / v41 )
          goto LABEL_21;
      }
      else if ( (*(_BYTE *)(v12 + 141) & 0x20) != 0 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          v55 = a1;
          v38 = a5;
          v39 = (_QWORD *)(a1 + 96);
          v40 = 2701;
LABEL_23:
          sub_6855B0(v40, v38, v39);
          sub_770D30(v55);
        }
        return 0;
      }
      v11 *= (_DWORD)v41;
      do
      {
        v12 = *(_QWORD *)(v12 + 160);
        v42 = *(_BYTE *)(v12 + 140);
      }
      while ( v42 == 12 );
      if ( v42 != 8 )
        goto LABEL_4;
    }
    while ( *(char *)(v12 + 168) >= 0 );
  }
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    v55 = a1;
    v38 = a5;
    v39 = (_QWORD *)(a1 + 96);
    v40 = 2999;
    goto LABEL_23;
  }
  return 0;
}

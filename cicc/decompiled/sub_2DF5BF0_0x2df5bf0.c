// Function: sub_2DF5BF0
// Address: 0x2df5bf0
//
void __fastcall sub_2DF5BF0(__int64 a1, int *a2, __int64 a3, char a4, char a5, __int64 a6)
{
  char v6; // r8
  _DWORD *v7; // r11
  int *v8; // r12
  char v10; // cl
  int *v11; // r15
  __int64 v12; // r8
  __int64 v13; // rax
  int v14; // r14d
  __int64 v15; // rsi
  _DWORD *v16; // rax
  unsigned __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r11
  char v20; // al
  __int64 *v21; // rdi
  __int64 v22; // r13
  __int64 v23; // r12
  _QWORD *v24; // rax
  _DWORD *v25; // rdx
  unsigned __int64 v26; // rdi
  _BYTE *v27; // r12
  char v28; // al
  unsigned int v29; // r12d
  _QWORD *v30; // rax
  void *v31; // r9
  unsigned __int64 v32; // rdi
  int v34; // [rsp+10h] [rbp-B0h]
  __int64 v35; // [rsp+30h] [rbp-90h] BYREF
  __int64 v36; // [rsp+38h] [rbp-88h]
  char v37; // [rsp+40h] [rbp-80h]
  void *src; // [rsp+50h] [rbp-70h] BYREF
  __int64 v39; // [rsp+58h] [rbp-68h]
  _BYTE v40[96]; // [rsp+60h] [rbp-60h] BYREF

  v6 = (a4 << 6) | (a5 << 7);
  v7 = v40;
  v8 = &a2[a3];
  v10 = *(_BYTE *)(a1 + 8);
  *(_QWORD *)(a1 + 16) = a6;
  *(_QWORD *)a1 = 0;
  *(_BYTE *)(a1 + 8) = v10 & 0x3F | v6;
  src = v40;
  v39 = 0xC00000000LL;
  if ( v8 == a2 )
  {
    LODWORD(v12) = 0;
    goto LABEL_27;
  }
  v11 = a2;
  v12 = 0;
  while ( 1 )
  {
    v14 = *v11;
    v15 = (__int64)&v7[v12];
    LODWORD(v35) = *v11;
    v16 = sub_2DF4E20(v7, v15, (int *)&v35);
    if ( (_DWORD *)v15 == v16 )
      break;
    ++v11;
    v13 = sub_B0D640(*(_QWORD **)(a1 + 16), v17, (unsigned int)(((__int64)v16 - v19) >> 2));
    v12 = (unsigned int)v39;
    *(_QWORD *)(a1 + 16) = v13;
    if ( v8 == v11 )
      goto LABEL_9;
LABEL_4:
    v7 = src;
  }
  if ( v17 + 1 > HIDWORD(v39) )
  {
    sub_C8D5F0((__int64)&src, v40, v17 + 1, 4u, v17, v18);
    v16 = (char *)src + 4 * (unsigned int)v39;
  }
  *v16 = v14;
  ++v11;
  v12 = (unsigned int)(v39 + 1);
  LODWORD(v39) = v39 + 1;
  if ( v8 != v11 )
    goto LABEL_4;
LABEL_9:
  if ( (unsigned int)v12 > 0x3F )
  {
    v35 = 4101;
    v20 = *(_BYTE *)(a1 + 8);
    v36 = 0;
    *(_BYTE *)(a1 + 8) = v20 & 0xC0 | 1;
    v21 = (__int64 *)(*(_QWORD *)(a6 + 8) & 0xFFFFFFFFFFFFFFF8LL);
    if ( (*(_QWORD *)(a6 + 8) & 4) != 0 )
      v21 = (__int64 *)*v21;
    *(_QWORD *)(a1 + 16) = sub_B0D000(v21, &v35, 2, 0, 1);
    sub_AF47B0((__int64)&v35, *(unsigned __int64 **)(a6 + 16), *(unsigned __int64 **)(a6 + 24));
    if ( v37 )
      *(_QWORD *)(a1 + 16) = sub_B0E470(*(_QWORD *)(a1 + 16), v36, v35);
    v22 = *(_BYTE *)(a1 + 8) & 0x3F;
    v23 = 4 * v22;
    v24 = (_QWORD *)sub_2207820(4 * v22);
    v25 = v24;
    if ( v24 && v22 )
    {
      if ( (unsigned int)v23 >= 8 )
      {
        *v24 = 0;
        *(_QWORD *)((char *)v24 + v23 - 8) = 0;
        memset(
          (void *)((unsigned __int64)(v24 + 1) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((unsigned int)v23 + (_DWORD)v24 - (((_DWORD)v24 + 8) & 0xFFFFFFF8)) >> 3));
        v26 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v24;
        if ( !v26 )
          goto LABEL_21;
        goto LABEL_20;
      }
      if ( (v23 & 4) == 0 )
      {
        if ( !(_DWORD)v23 )
          goto LABEL_19;
        *(_BYTE *)v24 = 0;
        v26 = *(_QWORD *)a1;
        *(_QWORD *)a1 = v24;
        if ( !v26 )
        {
LABEL_21:
          *v25 = -1;
          goto LABEL_22;
        }
LABEL_20:
        j_j___libc_free_0_0(v26);
        v25 = *(_DWORD **)a1;
        goto LABEL_21;
      }
      *(_DWORD *)v24 = 0;
      *(_DWORD *)((char *)v24 + v23 - 4) = 0;
    }
LABEL_19:
    v26 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v24;
    if ( !v26 )
      goto LABEL_21;
    goto LABEL_20;
  }
LABEL_27:
  v28 = v12 & 0x3F | *(_BYTE *)(a1 + 8) & 0xC0;
  *(_BYTE *)(a1 + 8) = v28;
  if ( (v28 & 0x3F) != 0 )
  {
    v34 = v12;
    v29 = 4 * v12;
    v30 = (_QWORD *)sub_2207820(4LL * (unsigned int)v12);
    v31 = v30;
    if ( v30 && v34 )
    {
      if ( v29 >= 8 )
      {
        *v30 = 0;
        *(_QWORD *)((char *)v30 + v29 - 8) = 0;
        memset(
          (void *)((unsigned __int64)(v30 + 1) & 0xFFFFFFFFFFFFFFF8LL),
          0,
          8LL * (((_DWORD)v30 - (((_DWORD)v30 + 8) & 0xFFFFFFF8) + v29) >> 3));
      }
      else if ( (v29 & 4) != 0 )
      {
        *(_DWORD *)v30 = 0;
        *(_DWORD *)((char *)v30 + v29 - 4) = 0;
      }
      else if ( v29 )
      {
        *(_BYTE *)v30 = 0;
      }
    }
    v32 = *(_QWORD *)a1;
    *(_QWORD *)a1 = v30;
    if ( v32 )
    {
      j_j___libc_free_0_0(v32);
      v31 = *(void **)a1;
    }
    v27 = src;
    if ( 4LL * (unsigned int)v39 )
      memmove(v31, src, 4LL * (unsigned int)v39);
  }
  else
  {
LABEL_22:
    v27 = src;
  }
  if ( v27 != v40 )
    _libc_free((unsigned __int64)v27);
}

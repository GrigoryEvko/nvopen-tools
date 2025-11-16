// Function: sub_5F6FE0
// Address: 0x5f6fe0
//
__int64 __fastcall sub_5F6FE0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 *a3,
        unsigned int a4,
        int a5,
        char a6,
        _QWORD *a7,
        _DWORD *a8,
        unsigned int a9)
{
  __int64 v14; // rbx
  unsigned __int64 v15; // rsi
  unsigned int v16; // eax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 ***v19; // r8
  __int64 **v20; // rax
  __int64 v21; // rbx
  __int64 v22; // rdi
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int64 *v25; // rdx
  char v26; // r9
  unsigned __int8 v28; // r9
  __int64 v29; // rax
  char v30; // r9
  unsigned int v31; // [rsp+0h] [rbp-50h]
  __int64 ***v32; // [rsp+0h] [rbp-50h]
  __int64 ***v33; // [rsp+0h] [rbp-50h]
  unsigned int v35; // [rsp+Ch] [rbp-44h]
  __int64 ***v36; // [rsp+18h] [rbp-38h] BYREF

  v14 = (int)a4;
  v15 = a4;
  v35 = unk_4F07270;
  v16 = sub_6935B0(a2, a4, &v36);
  v19 = v36;
  v31 = v16;
  if ( v36 )
  {
    v15 = a2;
    v20 = sub_5EA830(v36, (__int64 *)a2, a3);
    v18 = v31;
    v19 = (__int64 ***)v20;
    if ( !v20 )
    {
      v30 = *((_BYTE *)v36 + 24);
      if ( (v30 & 0x10) != 0 )
      {
        v15 = a2;
        v19 = (__int64 ***)sub_5F6FE0(
                             (_DWORD)v36,
                             a2,
                             (_DWORD)a3,
                             v31,
                             1,
                             (v30 & 0x20) != 0,
                             (__int64)a7,
                             (__int64)a8,
                             a9 | ((*(_BYTE *)(a1 + 25) & 0x40) != 0));
      }
      else
      {
        *a8 = 1;
      }
    }
  }
  else if ( a3 )
  {
    v15 = (unsigned __int64)a7;
    v33 = v36;
    sub_6854C0(2644, a7, *a3);
    v19 = v33;
  }
  v21 = qword_4F04C68[0] + 776 * v14;
  if ( *(_BYTE *)(v21 + 4) == 7 )
    v22 = unk_4F073B8;
  else
    v22 = *(unsigned int *)(v21 + 192);
  v32 = v19;
  sub_7296B0(v22, v15, v17, v18);
  v23 = sub_7275F0();
  v24 = *(__int64 **)a1;
  if ( *(_QWORD *)a1 )
  {
    do
    {
      v25 = v24;
      v24 = (__int64 *)*v24;
    }
    while ( v24 );
    *v25 = v23;
    if ( a3 )
    {
LABEL_8:
      *(_WORD *)(v23 + 32) |= 0x202u;
      *(_QWORD *)(v23 + 8) = a3;
      if ( !v32 )
        goto LABEL_10;
      goto LABEL_9;
    }
  }
  else
  {
    *(_QWORD *)a1 = v23;
    if ( a3 )
      goto LABEL_8;
  }
  *(_BYTE *)(v23 + 33) |= 2u;
  *(_QWORD *)(v23 + 8) = a2;
  if ( v32 )
  {
LABEL_9:
    *(_QWORD *)(v23 + 16) = v32;
    goto LABEL_10;
  }
  if ( !a2 && !*a8 )
    *(_BYTE *)(v23 + 32) |= 4u;
LABEL_10:
  v26 = *(_BYTE *)(v23 + 32) & 0xE7 | ((8 * a6) | (16 * a5)) & 0x18;
  *(_BYTE *)(v23 + 32) = v26;
  *(_QWORD *)(v23 + 36) = *a7;
  if ( !a5 )
    goto LABEL_11;
  if ( (v26 & 8) == 0 )
  {
    if ( (*(_BYTE *)(a1 + 24) & 2) != 0 )
      goto LABEL_20;
    goto LABEL_19;
  }
  v29 = *(_QWORD *)(v23 + 16);
  if ( v29 && (*(_BYTE *)(v29 + 33) & 4) != 0 )
LABEL_19:
    *(_BYTE *)(v23 + 33) |= 4u;
LABEL_20:
  v28 = v26 & 4;
  if ( !a2 )
  {
    if ( v28 )
      *(_BYTE *)(v23 + 32) |= 8u;
    goto LABEL_25;
  }
  if ( v28 | *(_BYTE *)(a2 + 172) & 1 )
    *(_BYTE *)(v23 + 32) |= 8u;
  if ( (*(_BYTE *)(a2 + 176) & 2) == 0 || (*(_BYTE *)(a1 + 25) & 8) != 0 || (*(_BYTE *)(a1 + 25) & 0x10) != 0 )
  {
LABEL_25:
    *(_QWORD *)(v23 + 24) = sub_5F6E90(a1, v23);
LABEL_11:
    if ( (*(_BYTE *)(a1 + 25) & 8) != 0 || (*(_BYTE *)(a1 + 25) & 0x10) != 0 )
      sub_5E5F40(v23, (*(_BYTE *)(a1 + 25) & 8) != 0, a9);
  }
  sub_729730(v35);
  return v23;
}

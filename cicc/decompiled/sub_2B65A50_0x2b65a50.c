// Function: sub_2B65A50
// Address: 0x2b65a50
//
__int64 __fastcall sub_2B65A50(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 *a4,
        __int64 a5,
        int a6,
        void *src,
        __int64 a8)
{
  int v8; // r15d
  char v11; // al
  char v12; // dl
  unsigned __int64 v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r14
  char v18; // al
  unsigned int v19; // r10d
  unsigned int v20; // r13d
  unsigned int v21; // r15d
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // r12
  _DWORD *v28; // rax
  _DWORD *v29; // rdx
  __int64 v30; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rsi
  int v35; // eax
  unsigned __int64 v36; // rax
  int *v37; // rdi
  __int64 v38; // [rsp-8h] [rbp-F8h]
  unsigned int v39; // [rsp+14h] [rbp-DCh]
  int v40; // [rsp+18h] [rbp-D8h]
  int v41; // [rsp+20h] [rbp-D0h]
  char v42; // [rsp+27h] [rbp-C9h]
  __int64 v43; // [rsp+28h] [rbp-C8h]
  unsigned int v44; // [rsp+38h] [rbp-B8h]
  int v45; // [rsp+3Ch] [rbp-B4h]
  unsigned int v46; // [rsp+4Ch] [rbp-A4h] BYREF
  char v47[32]; // [rsp+50h] [rbp-A0h] BYREF
  _BYTE *v48; // [rsp+70h] [rbp-80h] BYREF
  __int64 v49; // [rsp+78h] [rbp-78h]
  _BYTE v50[24]; // [rsp+80h] [rbp-70h] BYREF
  int v51; // [rsp+98h] [rbp-58h] BYREF
  unsigned __int64 v52; // [rsp+A0h] [rbp-50h]
  int *v53; // [rsp+A8h] [rbp-48h]
  int *v54; // [rsp+B0h] [rbp-40h]
  __int64 v55; // [rsp+B8h] [rbp-38h]

  v8 = a1;
  v44 = sub_2B65070(a1, a2, a3, a4, a5, a6, src, a8);
  v11 = *(_BYTE *)a2;
  if ( *(_BYTE *)a2 <= 0x1Cu )
    return v44;
  v12 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu || a2 == a3 || *(_DWORD *)(a1 + 36) == a6 || !v44 )
    return v44;
  if ( v11 == 61 )
  {
    if ( v12 == 61 )
      return v44;
    v39 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( v39 > 2 && (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) > 2 )
      return v44;
  }
  else
  {
    v39 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
    if ( v39 > 2 && (*(_DWORD *)(a3 + 4) & 0x7FFFFFFu) > 2 || v11 == 90 && v12 == 90 )
      return v44;
  }
  v51 = 0;
  v13 = 0;
  v48 = v50;
  v49 = 0x400000000LL;
  v52 = 0;
  v53 = &v51;
  v54 = &v51;
  v55 = 0;
  if ( v39 )
  {
    v40 = v8;
    v41 = a6 + 1;
    v14 = 0;
    v15 = a3;
    v16 = a2;
    v17 = v15;
    while ( 1 )
    {
      v46 = 0;
      v18 = sub_2B17690(v17);
      v19 = 0;
      if ( !v18 )
        v19 = v14;
      v20 = v14 + 1;
      v21 = v19;
      if ( sub_2B17690(v17) )
      {
        v20 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
      }
      else if ( (*(_DWORD *)(v17 + 4) & 0x7FFFFFFu) <= v20 )
      {
        v20 = *(_DWORD *)(v17 + 4) & 0x7FFFFFF;
      }
      if ( v21 != v20 )
        break;
LABEL_27:
      if ( v39 == (_DWORD)++v14 )
      {
        v13 = v52;
        goto LABEL_29;
      }
    }
    v42 = 0;
    v45 = 0;
    v43 = 32 * v14;
    v24 = v17;
    v25 = v14;
    v26 = v16;
    v27 = v24;
    while ( v55 )
    {
      v36 = v52;
      if ( !v52 )
        goto LABEL_38;
      v37 = &v51;
      do
      {
        v22 = *(_QWORD *)(v36 + 16);
        if ( *(_DWORD *)(v36 + 32) < v21 )
        {
          v36 = *(_QWORD *)(v36 + 24);
        }
        else
        {
          v37 = (int *)v36;
          v36 = *(_QWORD *)(v36 + 16);
        }
      }
      while ( v36 );
      if ( v37 == &v51 || v37[8] > v21 )
        goto LABEL_38;
LABEL_24:
      if ( ++v21 == v20 )
      {
        v30 = v27;
        v16 = v26;
        v14 = v25;
        v17 = v30;
        if ( v42 )
        {
          sub_2B5C0F0((__int64)v47, (__int64)&v48, &v46, v22, v23);
          v44 += v45;
        }
        goto LABEL_27;
      }
    }
    v28 = v48;
    v29 = &v48[4 * (unsigned int)v49];
    if ( v48 != (_BYTE *)v29 )
    {
      while ( *v28 != v21 )
      {
        if ( v29 == ++v28 )
          goto LABEL_38;
      }
      if ( v29 != v28 )
        goto LABEL_24;
    }
LABEL_38:
    if ( (*(_BYTE *)(v27 + 7) & 0x40) != 0 )
      v32 = *(_QWORD *)(v27 - 8);
    else
      v32 = v27 - 32LL * (*(_DWORD *)(v27 + 4) & 0x7FFFFFF);
    v33 = *(_QWORD *)(v32 + 32LL * v21);
    if ( (*(_BYTE *)(v26 + 7) & 0x40) != 0 )
      v34 = *(_QWORD *)(v26 - 8);
    else
      v34 = v26 - 32LL * (*(_DWORD *)(v26 + 4) & 0x7FFFFFF);
    v35 = sub_2B65A50(v40, *(_QWORD *)(v34 + v43), v33, v26, v27, v41, 0, 0);
    v22 = v38;
    if ( v35 > v45 )
    {
      v46 = v21;
      v45 = v35;
      v42 = 1;
    }
    goto LABEL_24;
  }
LABEL_29:
  sub_2B10A80(v13);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v44;
}

// Function: sub_2B330C0
// Address: 0x2b330c0
//
__int64 __fastcall sub_2B330C0(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        unsigned int a4,
        __int64 (__fastcall *a5)(__int64, _QWORD *, __int64, void *, _QWORD),
        __int64 a6)
{
  __int64 v9; // rax
  unsigned int v10; // r11d
  __int64 v11; // rax
  unsigned __int64 v14; // r8
  int v15; // r12d
  __int64 v16; // r9
  _QWORD *v17; // rsi
  __int64 v18; // rax
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax
  int v22; // esi
  __int64 result; // rax
  unsigned int v24; // eax
  unsigned int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  __int64 v33; // r12
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 **v36; // rdi
  _BYTE *v37; // rax
  _BYTE *v38; // rax
  __int64 v39; // [rsp+0h] [rbp-120h]
  unsigned int v40; // [rsp+Ch] [rbp-114h]
  unsigned __int64 v41; // [rsp+10h] [rbp-110h]
  __int64 na; // [rsp+18h] [rbp-108h]
  size_t n; // [rsp+18h] [rbp-108h]
  size_t nb; // [rsp+18h] [rbp-108h]
  __int64 v46; // [rsp+20h] [rbp-100h]
  __int64 v47; // [rsp+28h] [rbp-F8h]
  __int64 v48; // [rsp+38h] [rbp-E8h]
  _QWORD v49[4]; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v50; // [rsp+60h] [rbp-C0h]
  void *s; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v52; // [rsp+78h] [rbp-A8h]
  _QWORD v53[6]; // [rsp+80h] [rbp-A0h] BYREF
  _BYTE *v54; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-68h]
  _BYTE v56[16]; // [rsp+C0h] [rbp-60h] BYREF
  __int16 v57; // [rsp+D0h] [rbp-50h]

  v9 = *(_QWORD *)(a3 + 8);
  if ( *(_BYTE *)(v9 + 8) != 17 || (v10 = *(_DWORD *)(v9 + 32), !(a4 % v10)) )
  {
    v57 = 257;
    v28 = sub_BCB2E0(*(_QWORD **)(a1 + 72));
    v29 = sub_ACD640(v28, a4, 0);
    v30 = a2[1];
    BYTE4(v48) = 0;
    v53[0] = v29;
    v31 = *(_QWORD *)(a3 + 8);
    v49[0] = v30;
    s = a2;
    v49[1] = v31;
    v52 = a3;
    return sub_B33D10(a1, 0x17Eu, (__int64)v49, 2, (int)&s, 3, v48, (__int64)&v54);
  }
  v11 = a2[1];
  if ( *(_BYTE *)(v11 + 8) != 17 )
  {
    HIDWORD(v52) = 12;
    v15 = 1;
    v16 = 4;
    v14 = 1;
    s = v53;
    v24 = 4;
    goto LABEL_18;
  }
  v14 = *(unsigned int *)(v11 + 32);
  s = v53;
  v15 = v14;
  v16 = 4 * v14;
  v52 = 0xC00000000LL;
  if ( v14 > 0xC )
  {
    v39 = a6;
    v40 = v10;
    v41 = v14;
    na = 4 * v14;
    sub_C8D5F0((__int64)&s, v53, v14, 4u, v14, v16);
    memset(s, 255, na);
    LODWORD(v52) = v15;
    v17 = s;
    v16 = na;
    v14 = v41;
    v10 = v40;
    a6 = v39;
    goto LABEL_8;
  }
  if ( v14 )
  {
    v24 = 4 * v14;
    if ( !v16 )
    {
      v17 = v53;
      goto LABEL_7;
    }
LABEL_18:
    if ( v24 < 8 )
    {
      if ( (v24 & 4) != 0 )
      {
        LODWORD(v53[0]) = -1;
        *(_DWORD *)((char *)&v52 + v24 + 4) = -1;
      }
      else if ( v24 )
      {
        LOBYTE(v53[0]) = -1;
        if ( (v24 & 2) != 0 )
          *(_WORD *)((char *)&v52 + v24 + 6) = -1;
      }
    }
    else
    {
      *(_QWORD *)((char *)&v53[-1] + v24) = -1;
      if ( v24 - 1 >= 8 )
      {
        v25 = (v24 - 1) & 0xFFFFFFF8;
        v26 = 0;
        do
        {
          v27 = v26;
          v26 += 8;
          *(_QWORD *)((char *)v53 + v27) = -1;
        }
        while ( v26 < v25 );
      }
    }
    v17 = s;
    goto LABEL_7;
  }
  v17 = v53;
  v15 = 0;
LABEL_7:
  LODWORD(v52) = v15;
LABEL_8:
  v18 = 0;
  v19 = (unsigned __int64)(v16 - 4) >> 2;
  if ( v16 )
  {
    do
    {
      v20 = v18;
      *((_DWORD *)v17 + v18) = v18;
      ++v18;
    }
    while ( v19 != v20 );
  }
  v21 = 0;
  if ( v10 )
  {
    do
    {
      v19 = a4 + (unsigned int)v21;
      v22 = v15 + v21++;
      *((_DWORD *)s + v19) = v22;
    }
    while ( v10 != v21 );
  }
  if ( a5 )
  {
    result = a5(a6, a2, a3, s, (unsigned int)v52);
  }
  else
  {
    n = v10;
    v54 = v56;
    v55 = 0xC00000000LL;
    sub_11B1960((__int64)&v54, v14, -1, v19, v14, v16);
    v32 = (unsigned __int64)v54;
    v33 = (__int64)&v54[4 * n];
    if ( v54 != (_BYTE *)v33 )
    {
      v34 = 0;
      do
      {
        v35 = v34;
        *(_DWORD *)(v32 + 4 * v34) = v34;
        ++v34;
      }
      while ( v35 != (4 * n - 4) >> 2 );
      v33 = (__int64)v54;
    }
    v36 = *(__int64 ***)(a3 + 8);
    v50 = 257;
    nb = (unsigned int)v55;
    v37 = (_BYTE *)sub_ACADE0(v36);
    v38 = (_BYTE *)sub_A83CB0((unsigned int **)a1, (_BYTE *)a3, v37, v33, nb, (__int64)v49);
    v50 = 257;
    result = sub_A83CB0((unsigned int **)a1, a2, v38, (__int64)s, (unsigned int)v52, (__int64)v49);
    if ( v54 != v56 )
    {
      v46 = result;
      _libc_free((unsigned __int64)v54);
      result = v46;
    }
  }
  if ( s != v53 )
  {
    v47 = result;
    _libc_free((unsigned __int64)s);
    return v47;
  }
  return result;
}

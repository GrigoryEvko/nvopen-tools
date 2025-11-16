// Function: sub_2CDCF50
// Address: 0x2cdcf50
//
__int64 __fastcall sub_2CDCF50(__int64 *a1, __int64 a2, __int64 a3, __int64 **a4, __int64 a5, __int64 a6, __int64 a7)
{
  unsigned int v7; // r14d
  __int64 v8; // rbx
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  unsigned __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  char v20; // [rsp+1Ch] [rbp-C4h]
  __int64 v22; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v23; // [rsp+38h] [rbp-A8h]
  __int64 v24; // [rsp+40h] [rbp-A0h]
  __int64 v25; // [rsp+48h] [rbp-98h]
  void *src; // [rsp+50h] [rbp-90h] BYREF
  unsigned __int64 v27; // [rsp+58h] [rbp-88h]
  __int64 v28; // [rsp+60h] [rbp-80h] BYREF
  int v29; // [rsp+68h] [rbp-78h]
  char v30; // [rsp+6Ch] [rbp-74h]
  char v31; // [rsp+70h] [rbp-70h] BYREF

  a1[2] = a6;
  *a1 = a2;
  v7 = (unsigned __int8)qword_5013C48;
  v20 = a3;
  if ( !(_BYTE)qword_5013C48 )
  {
LABEL_18:
    if ( (unsigned __int8)sub_CE9220(a2) )
    {
      src = 0;
      v27 = (unsigned __int64)&v31;
      v30 = 1;
      v28 = 8;
      v29 = 0;
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      sub_2CDA660((__int64)a1, (__int64)&src, (__int64)&v22, v15, v16, v17);
      v7 |= sub_2CD9C00(a1, (__int64)&src, (__int64)&v22);
      sub_C7D6A0(v23, 8LL * (unsigned int)v25, 8);
      if ( !v30 )
        _libc_free(v27);
    }
    return v7;
  }
  v8 = *(_QWORD *)(a2 + 80);
  if ( !v8 )
  {
    src = &v28;
    v27 = 0x400000000LL;
    BUG();
  }
  src = &v28;
  v9 = v8 + 24;
  v27 = 0x400000000LL;
  v10 = *(_QWORD *)(v9 + 8);
  if ( v10 == v9 )
    goto LABEL_14;
  do
  {
    if ( !v10 )
      BUG();
    if ( *(_BYTE *)(v10 - 24) == 60 )
    {
      v22 = 0;
      v23 = 0;
      v24 = 0;
      v25 = 0;
      if ( (unsigned __int8)sub_2A4FF40(v10 - 24, (__int64)&v22, a3, (__int64)a4, a5, a6) )
      {
        v13 = (unsigned int)v27;
        v14 = (unsigned int)v27 + 1LL;
        if ( v14 > HIDWORD(v27) )
        {
          sub_C8D5F0((__int64)&src, &v28, v14, 8u, v11, v12);
          v13 = (unsigned int)v27;
        }
        *((_QWORD *)src + v13) = v10 - 24;
        LODWORD(v27) = v27 + 1;
      }
      sub_C7D6A0(v23, 8LL * (unsigned int)v25, 8);
    }
    v10 = *(_QWORD *)(v10 + 8);
  }
  while ( v9 != v10 );
  if ( (_DWORD)v27 )
    sub_2A57B70(src, (unsigned int)v27, a4, 0, 1);
  else
LABEL_14:
    v7 = 0;
  v7 |= sub_31CCB00(a4, a5, a7);
  if ( v20 )
  {
    if ( src != &v28 )
      _libc_free((unsigned __int64)src);
    goto LABEL_18;
  }
  if ( src != &v28 )
    _libc_free((unsigned __int64)src);
  return v7;
}

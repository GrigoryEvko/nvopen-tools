// Function: sub_2721630
// Address: 0x2721630
//
_QWORD *__fastcall sub_2721630(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v14; // rdx
  __int64 i; // rbx
  __int64 v16; // rcx
  __int64 *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 *v21; // rax
  __int64 *v22; // rax
  __int64 *v23; // rdi
  int v24; // ebx
  __int64 v25; // [rsp+10h] [rbp-3B0h]
  _BYTE *v26; // [rsp+18h] [rbp-3A8h] BYREF
  __int64 v27; // [rsp+20h] [rbp-3A0h]
  _BYTE v28[200]; // [rsp+28h] [rbp-398h] BYREF
  __int64 v29; // [rsp+F0h] [rbp-2D0h]
  _BYTE *v30; // [rsp+F8h] [rbp-2C8h] BYREF
  __int64 v31; // [rsp+100h] [rbp-2C0h]
  _BYTE v32[200]; // [rsp+108h] [rbp-2B8h] BYREF
  unsigned __int64 v33; // [rsp+1D0h] [rbp-1F0h] BYREF
  _BYTE *v34; // [rsp+1D8h] [rbp-1E8h] BYREF
  __int64 v35; // [rsp+1E0h] [rbp-1E0h]
  _BYTE v36[200]; // [rsp+1E8h] [rbp-1D8h] BYREF
  __int64 v37; // [rsp+2B0h] [rbp-110h] BYREF
  __int64 *v38; // [rsp+2B8h] [rbp-108h] BYREF
  __int64 v39; // [rsp+2C0h] [rbp-100h]
  _BYTE v40[248]; // [rsp+2C8h] [rbp-F8h] BYREF

  v7 = a3;
  v8 = *(_BYTE *)(a3 + 28) == 0;
  v29 = a3;
  v30 = v32;
  v9 = *a2;
  v31 = 0x800000000LL;
  v37 = a3;
  v38 = (__int64 *)v40;
  v39 = 0x800000000LL;
  if ( v8 )
    goto LABEL_25;
  v10 = *(_QWORD **)(a3 + 8);
  a4 = *(unsigned int *)(a3 + 20);
  a3 = (__int64)&v10[a4];
  if ( v10 == (_QWORD *)a3 )
  {
LABEL_24:
    if ( (unsigned int)a4 < *(_DWORD *)(v7 + 16) )
    {
      *(_DWORD *)(v7 + 20) = a4 + 1;
      *(_QWORD *)a3 = v9;
      LODWORD(a3) = v39;
      ++*(_QWORD *)v7;
      goto LABEL_29;
    }
LABEL_25:
    a2 = (__int64 *)v9;
    sub_C8CC70(v7, v9, a3, a4, a5, a6);
    a5 = v14;
    a3 = (unsigned int)v39;
    if ( !(_BYTE)a5 )
    {
LABEL_26:
      v26 = v28;
      v25 = v37;
      v27 = 0x800000000LL;
      if ( (_DWORD)a3 )
        sub_27214F0((__int64)&v26, (__int64 *)&v38, a3, a4, a5, a6);
      goto LABEL_7;
    }
LABEL_29:
    for ( i = *(_QWORD *)(v9 + 16); i; i = *(_QWORD *)(i + 8) )
    {
      if ( (unsigned __int8)(**(_BYTE **)(i + 24) - 30) <= 0xAu )
        break;
    }
    if ( HIDWORD(v39) <= (unsigned int)a3 )
    {
      a2 = (__int64 *)sub_C8D7D0((__int64)&v38, (__int64)v40, 0, 0x18u, &v33, a6);
      v20 = 3LL * (unsigned int)v39;
      v21 = &a2[v20];
      if ( &a2[v20] )
      {
        *v21 = 0;
        v21[1] = i;
        v21[2] = v9;
        v20 = 3LL * (unsigned int)v39;
      }
      v22 = v38;
      v23 = &v38[v20];
      if ( v38 != v23 )
      {
        v18 = (__int64)a2;
        do
        {
          if ( v18 )
          {
            *(_QWORD *)v18 = *v22;
            *(_QWORD *)(v18 + 8) = v22[1];
            v16 = v22[2];
            *(_QWORD *)(v18 + 16) = v16;
          }
          v22 += 3;
          v18 += 24;
        }
        while ( v23 != v22 );
        v23 = v38;
      }
      v24 = v33;
      if ( v23 != (__int64 *)v40 )
        _libc_free((unsigned __int64)v23);
      LODWORD(v39) = v39 + 1;
      v38 = a2;
      HIDWORD(v39) = v24;
    }
    else
    {
      v16 = 3LL * (unsigned int)a3;
      v17 = &v38[3 * (unsigned int)a3];
      if ( v17 )
      {
        *v17 = 0;
        v17[1] = i;
        v17[2] = v9;
        LODWORD(a3) = v39;
      }
      v18 = (unsigned int)(a3 + 1);
      LODWORD(v39) = v18;
    }
    sub_2720090((__int64)&v37, (__int64)a2, v18, v16, a5, a6);
    a3 = (unsigned int)v39;
    goto LABEL_26;
  }
  while ( v9 != *v10 )
  {
    if ( (_QWORD *)a3 == ++v10 )
      goto LABEL_24;
  }
  v26 = v28;
  v25 = v37;
  v27 = 0x800000000LL;
LABEL_7:
  if ( v38 != (__int64 *)v40 )
    _libc_free((unsigned __int64)v38);
  v38 = (__int64 *)v40;
  v37 = v29;
  v39 = 0x800000000LL;
  if ( (_DWORD)v31 )
    sub_2721330((__int64)&v38, (__int64)&v30, a3, a4, a5, a6);
  v11 = (unsigned int)v27;
  v34 = v36;
  v33 = v25;
  v35 = 0x800000000LL;
  if ( (_DWORD)v27 )
  {
    sub_2721330((__int64)&v34, (__int64)&v26, v25, (unsigned int)v27, (__int64)&v34, a6);
    v19 = v33;
    a1[2] = 0x800000000LL;
    *a1 = v19;
    a1[1] = a1 + 3;
    v12 = (unsigned int)v35;
    if ( (_DWORD)v35 )
      sub_2721330((__int64)(a1 + 1), (__int64)&v34, (unsigned int)v35, v11, (__int64)&v34, a6);
  }
  else
  {
    *a1 = v25;
    v12 = (__int64)(a1 + 3);
    a1[1] = a1 + 3;
    a1[2] = 0x800000000LL;
  }
  a1[27] = v37;
  a1[28] = a1 + 30;
  a1[29] = 0x800000000LL;
  if ( (_DWORD)v39 )
    sub_2721330((__int64)(a1 + 28), (__int64)&v38, v12, v11, a5, a6);
  if ( v34 != v36 )
    _libc_free((unsigned __int64)v34);
  if ( v38 != (__int64 *)v40 )
    _libc_free((unsigned __int64)v38);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  return a1;
}

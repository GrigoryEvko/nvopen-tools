// Function: sub_2DDE490
// Address: 0x2dde490
//
void __fastcall sub_2DDE490(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  unsigned __int64 v9; // r12
  __int64 v10; // rdx
  unsigned int *v11; // rdi
  unsigned __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int **v14; // rsi
  unsigned int *v15; // rcx
  unsigned int v16; // eax
  unsigned int *v17; // r9
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdi
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r13
  __int64 v25; // rcx
  __int64 v26; // r15
  _DWORD *v27; // rsi
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rdx
  __int64 v35; // [rsp+0h] [rbp-B0h]
  __int64 i; // [rsp+10h] [rbp-A0h]
  __int64 v37; // [rsp+18h] [rbp-98h]
  _BYTE *v38; // [rsp+20h] [rbp-90h] BYREF
  __int64 v39; // [rsp+28h] [rbp-88h]
  _BYTE v40[32]; // [rsp+30h] [rbp-80h] BYREF
  _BYTE *v41; // [rsp+50h] [rbp-60h] BYREF
  __int64 v42; // [rsp+58h] [rbp-58h]
  _BYTE v43[80]; // [rsp+60h] [rbp-50h] BYREF

  v6 = a2 - (_QWORD)a1;
  v37 = a3;
  if ( a2 - (__int64)a1 <= 768 )
    return;
  v9 = a2;
  v10 = v6;
  if ( !a3 )
  {
    v26 = a2;
    goto LABEL_38;
  }
  while ( 2 )
  {
    v11 = a1[6];
    --v37;
    v12 = ((0xAAAAAAAAAAAAAAABLL * (v10 >> 4)) & 0xFFFFFFFFFFFFFFFELL)
        + ((__int64)(0xAAAAAAAAAAAAAAABLL * (v10 >> 4)) >> 1);
    v13 = *v11;
    v14 = &a1[2 * v12];
    v15 = *v14;
    v16 = **v14;
    if ( (unsigned int)v13 >= v16 && ((_DWORD)v13 != v16 || v11[1] >= v15[1]) )
    {
      v17 = *(unsigned int **)(v9 - 48);
      v18 = *v17;
      if ( (unsigned int)v13 >= (unsigned int)v18 )
      {
        if ( (_DWORD)v13 != (_DWORD)v18 || (v13 = v17[1], v11[1] >= (unsigned int)v13) )
        {
          if ( v16 >= (unsigned int)v18 && (v16 != (_DWORD)v18 || v15[1] >= v17[1]) )
            goto LABEL_9;
LABEL_32:
          sub_2DDE290((__int64)a1, v9 - 48, v13, (__int64)v15, v18, (__int64)v17);
          goto LABEL_10;
        }
      }
LABEL_26:
      sub_2DDE290((__int64)a1, (__int64)(a1 + 6), v13, (__int64)v15, v18, (__int64)v17);
      goto LABEL_10;
    }
    v17 = *(unsigned int **)(v9 - 48);
    v18 = *v17;
    if ( v16 >= (unsigned int)v18 && (v16 != (_DWORD)v18 || v15[1] >= v17[1]) )
    {
      if ( (unsigned int)v13 < (unsigned int)v18 || (_DWORD)v13 == (_DWORD)v18 && v11[1] < v17[1] )
        goto LABEL_32;
      goto LABEL_26;
    }
LABEL_9:
    sub_2DDE290((__int64)a1, (__int64)v14, v13, (__int64)v15, v18, (__int64)v17);
LABEL_10:
    v22 = (__int64)*a1;
    v23 = (unsigned __int64)(a1 + 6);
    v24 = v9;
    v25 = **a1;
    while ( 1 )
    {
      v26 = v23;
      if ( **(_DWORD **)v23 >= (unsigned int)v25
        && (**(_DWORD **)v23 != (_DWORD)v25 || *(_DWORD *)(*(_QWORD *)v23 + 4LL) >= *(_DWORD *)(v22 + 4)) )
      {
        break;
      }
LABEL_12:
      v23 += 48LL;
    }
    do
    {
      do
      {
        v27 = *(_DWORD **)(v24 - 48);
        v24 -= 48LL;
      }
      while ( *v27 > (unsigned int)v25 );
    }
    while ( *v27 == (_DWORD)v25 && *(_DWORD *)(v22 + 4) < v27[1] );
    if ( v23 < v24 )
    {
      sub_2DDE290(v23, v24, v19, v25, v20, v21);
      v22 = (__int64)*a1;
      v25 = **a1;
      goto LABEL_12;
    }
    sub_2DDE490(v23, v9, v37);
    v10 = v23 - (_QWORD)a1;
    if ( (__int64)(v23 - (_QWORD)a1) > 768 )
    {
      if ( v37 )
      {
        v9 = v23;
        continue;
      }
LABEL_38:
      v35 = 0xAAAAAAAAAAAAAAABLL * (v10 >> 4);
      v28 = (v35 - 2) >> 1;
      for ( i = (__int64)&(&a1[2 * v28])[2 * ((v35 - 2) & 0xFFFFFFFFFFFFFFFELL)]; ; i -= 48 )
      {
        v38 = v40;
        v39 = 0x400000000LL;
        if ( *(_DWORD *)(i + 8) )
        {
          sub_2DDB710((__int64)&v38, i, i, a4, a5, a6);
          v41 = v43;
          v42 = 0x400000000LL;
          if ( (_DWORD)v39 )
            sub_2DDB710((__int64)&v41, (__int64)&v38, v29, (unsigned int)v39, a5, a6);
        }
        else
        {
          v42 = 0x400000000LL;
          v41 = v43;
        }
        sub_2DDB890((__int64)a1, v28, v35, &v41, a5, a6);
        if ( v41 != v43 )
          _libc_free((unsigned __int64)v41);
        if ( !v28 )
          break;
        --v28;
        if ( v38 != v40 )
          _libc_free((unsigned __int64)v38);
      }
      if ( v38 != v40 )
        _libc_free((unsigned __int64)v38);
      do
      {
        v34 = *(unsigned int *)(v26 - 40);
        v26 -= 48;
        v38 = v40;
        v39 = 0x400000000LL;
        if ( (_DWORD)v34 )
          sub_2DDB710((__int64)&v38, v26, v34, a4, a5, a6);
        sub_2DDB710(v26, (__int64)a1, v34, a4, a5, a6);
        v42 = 0x400000000LL;
        v41 = v43;
        if ( (_DWORD)v39 )
          sub_2DDB710((__int64)&v41, (__int64)&v38, v30, v31, v32, v33);
        sub_2DDB890((__int64)a1, 0, 0xAAAAAAAAAAAAAAABLL * ((v26 - (__int64)a1) >> 4), &v41, v32, v33);
        if ( v41 != v43 )
          _libc_free((unsigned __int64)v41);
        if ( v38 != v40 )
          _libc_free((unsigned __int64)v38);
      }
      while ( v26 - (__int64)a1 > 48 );
    }
    break;
  }
}

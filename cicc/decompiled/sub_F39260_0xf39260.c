// Function: sub_F39260
// Address: 0xf39260
//
__int64 __fastcall sub_F39260(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // r12
  __int64 v5; // r8
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // rcx
  char *v10; // r9
  unsigned __int64 v11; // rdx
  unsigned __int64 *v12; // rdi
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  _BYTE *v15; // r14
  __int64 v16; // rbx
  unsigned int v17; // r15d
  __int64 v18; // rcx
  _BYTE *v19; // rdi
  int v20; // eax
  _QWORD *v21; // rbx
  __int64 v22; // rax
  char *v24; // [rsp+8h] [rbp-138h]
  __int64 v25; // [rsp+18h] [rbp-128h]
  _QWORD v26[2]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v27; // [rsp+30h] [rbp-110h]
  unsigned __int64 v28; // [rsp+40h] [rbp-100h] BYREF
  __int64 v29; // [rsp+48h] [rbp-F8h]
  _BYTE v30[240]; // [rsp+50h] [rbp-F0h] BYREF

  v4 = a2;
  v28 = (unsigned __int64)v30;
  v29 = 0x800000000LL;
  v6 = sub_AA5930(a1);
  v8 = v7;
  while ( v8 != v6 )
  {
    v27 = v6;
    v26[0] = 6;
    v26[1] = 0;
    if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
      sub_BD73F0((__int64)v26);
    v9 = (unsigned int)v29;
    v10 = (char *)v26;
    a2 = v28;
    v11 = (unsigned int)v29;
    if ( (unsigned __int64)(unsigned int)v29 + 1 > HIDWORD(v29) )
    {
      if ( v28 > (unsigned __int64)v26 || (v11 = v28 + 24LL * (unsigned int)v29, (unsigned __int64)v26 >= v11) )
      {
        sub_F39130((__int64)&v28, (unsigned int)v29 + 1LL, v11, (unsigned int)v29, v5, (__int64)v26);
        v9 = (unsigned int)v29;
        a2 = v28;
        v10 = (char *)v26;
        LODWORD(v11) = v29;
      }
      else
      {
        v24 = (char *)v26 - v28;
        sub_F39130((__int64)&v28, (unsigned int)v29 + 1LL, v11, (unsigned int)v29, v5, (__int64)v26);
        a2 = v28;
        v9 = (unsigned int)v29;
        v10 = &v24[v28];
        LODWORD(v11) = v29;
      }
    }
    v12 = (unsigned __int64 *)(a2 + 24 * v9);
    if ( v12 )
    {
      *v12 = 6;
      v13 = *((_QWORD *)v10 + 2);
      v12[1] = 0;
      v12[2] = v13;
      if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
      {
        a2 = *(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL;
        sub_BD6050(v12, a2);
      }
      LODWORD(v11) = v29;
    }
    LODWORD(v29) = v11 + 1;
    LOBYTE(a2) = v27 != -4096;
    if ( ((v27 != 0) & (unsigned __int8)a2) != 0 && v27 != -8192 )
      sub_BD60C0(v26);
    if ( !v6 )
      BUG();
    v14 = *(_QWORD *)(v6 + 32);
    if ( !v14 )
      BUG();
    v6 = 0;
    if ( *(_BYTE *)(v14 - 24) == 84 )
      v6 = v14 - 24;
  }
  v15 = (_BYTE *)v28;
  if ( (_DWORD)v29 )
  {
    v16 = 0;
    v17 = 0;
    v18 = 24LL * (unsigned int)v29;
    do
    {
      while ( 1 )
      {
        v19 = *(_BYTE **)&v15[v16 + 16];
        if ( v19 )
        {
          if ( *v19 == 84 )
            break;
        }
        v16 += 24;
        if ( v18 == v16 )
          goto LABEL_25;
      }
      a2 = v4;
      v25 = v18;
      v16 += 24;
      v20 = sub_F5CB10(v19, v4, a3);
      v18 = v25;
      v15 = (_BYTE *)v28;
      v17 |= v20;
    }
    while ( v25 != v16 );
LABEL_25:
    v21 = &v15[24 * (unsigned int)v29];
    if ( v21 != (_QWORD *)v15 )
    {
      do
      {
        v22 = *(v21 - 1);
        v21 -= 3;
        if ( v22 != -4096 && v22 != 0 && v22 != -8192 )
          sub_BD60C0(v21);
      }
      while ( v21 != (_QWORD *)v15 );
      v15 = (_BYTE *)v28;
    }
  }
  else
  {
    v17 = 0;
  }
  if ( v15 != v30 )
    _libc_free(v15, a2);
  return v17;
}

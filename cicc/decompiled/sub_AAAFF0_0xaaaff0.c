// Function: sub_AAAFF0
// Address: 0xaaaff0
//
__int64 __fastcall sub_AAAFF0(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r12
  __int64 v7; // rcx
  int v8; // eax
  __int64 result; // rax
  __int64 v10; // rax
  unsigned __int8 *v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rsi
  int v14; // eax
  __int64 v15; // r13
  __int64 v16; // r12
  __int64 v17; // r14
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  _BYTE *v21; // rdi
  __int64 v22; // [rsp+8h] [rbp-F8h]
  __int64 v23; // [rsp+18h] [rbp-E8h]
  __int64 v24; // [rsp+18h] [rbp-E8h]
  _QWORD v25[4]; // [rsp+20h] [rbp-E0h] BYREF
  _BYTE *v26; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v27; // [rsp+48h] [rbp-B8h]
  _BYTE v28[176]; // [rsp+50h] [rbp-B0h] BYREF

  v6 = *((_QWORD *)a2 + 1);
  v7 = *a2;
  v8 = *(unsigned __int8 *)(v6 + 8);
  if ( v8 == 17 )
  {
    if ( (_BYTE)v7 != 18 )
      goto LABEL_18;
    if ( (_DWORD)a1 == 12 )
      goto LABEL_10;
    return 0;
  }
  a3 = (unsigned int)(unsigned __int8)v7 - 12;
  if ( (unsigned int)a3 > 1 )
  {
    if ( (_BYTE)v7 != 18 )
      goto LABEL_17;
    if ( (_DWORD)a1 == 12 )
    {
LABEL_10:
      v10 = sub_C33340(a1, a2, a3, v7, a5);
      v11 = a2 + 24;
      v12 = v10;
      if ( *((_QWORD *)a2 + 3) == v10 )
        sub_C3C790(v25, v11);
      else
        sub_C33EB0(v25, v11);
      if ( v25[0] == v12 )
        sub_C3CCB0(v25);
      else
        sub_C34440(v25);
      if ( v25[0] == v12 )
        sub_C3C840(&v26, v25);
      else
        sub_C338E0(&v26, v25);
      v23 = sub_AD8F10(*((_QWORD *)a2 + 1), &v26);
      sub_91D830(&v26);
      sub_91D830(v25);
      return v23;
    }
    return 0;
  }
  if ( (_DWORD)a1 == 12 )
    return (__int64)a2;
  if ( (_DWORD)a1 == 13 )
    BUG();
  if ( (_BYTE)v7 == 18 )
    return 0;
LABEL_17:
  if ( (unsigned int)(v8 - 17) > 1 )
    return 0;
LABEL_18:
  v13 = sub_AD7630(a2, 0);
  if ( v13 && sub_AAAFF0((unsigned int)a1, v13) )
  {
    v14 = *(_DWORD *)(v6 + 32);
    BYTE4(v26) = *(_BYTE *)(v6 + 8) == 18;
    LODWORD(v26) = v14;
    return sub_AD5E10((size_t)v26);
  }
  if ( *(_BYTE *)(v6 + 8) != 17 )
    return 0;
  v15 = sub_BCCE00(*(_QWORD *)v6, 32);
  v26 = v28;
  v27 = 0x1000000000LL;
  v16 = *(unsigned int *)(v6 + 32);
  if ( (_DWORD)v16 )
  {
    v17 = 0;
    while ( 1 )
    {
      v18 = sub_AD64C0(v15, v17, 0);
      v19 = sub_AD5840(a2, v18, 0);
      result = sub_AAAFF0((unsigned int)a1, v19);
      if ( !result )
        break;
      v20 = (unsigned int)v27;
      if ( (unsigned __int64)(unsigned int)v27 + 1 > HIDWORD(v27) )
      {
        v22 = result;
        sub_C8D5F0(&v26, v28, (unsigned int)v27 + 1LL, 8);
        v20 = (unsigned int)v27;
        result = v22;
      }
      ++v17;
      *(_QWORD *)&v26[8 * v20] = result;
      v19 = (unsigned int)(v27 + 1);
      LODWORD(v27) = v27 + 1;
      if ( v16 == v17 )
      {
        v21 = v26;
        goto LABEL_32;
      }
    }
  }
  else
  {
    v21 = v28;
    v19 = 0;
LABEL_32:
    result = sub_AD3730(v21, v19);
  }
  if ( v26 != v28 )
  {
    v24 = result;
    _libc_free(v26, v19);
    return v24;
  }
  return result;
}

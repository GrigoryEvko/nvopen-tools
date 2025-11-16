// Function: sub_AA9CA0
// Address: 0xaa9ca0
//
__int64 __fastcall sub_AA9CA0(unsigned __int8 *a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  int v8; // eax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned int v12; // r14d
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rax
  unsigned __int8 *v21; // r12
  unsigned __int8 *v22; // rdi
  unsigned __int8 v23; // al
  char v24; // al
  int v25; // eax
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-F0h]
  __int64 v31; // [rsp+18h] [rbp-D8h]
  __int64 v32; // [rsp+18h] [rbp-D8h]
  __int64 v33; // [rsp+20h] [rbp-D0h]
  unsigned __int8 *v34; // [rsp+28h] [rbp-C8h]
  _BYTE *v35; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v36; // [rsp+38h] [rbp-B8h]
  _BYTE v37[176]; // [rsp+40h] [rbp-B0h] BYREF

  v34 = (unsigned __int8 *)a2;
  if ( (unsigned __int8)sub_AC30F0(a1) )
    return (__int64)a3;
  if ( (unsigned __int8)sub_AD7930(a1) )
    return (__int64)v34;
  v8 = *a1;
  if ( (_BYTE)v8 != 11 )
  {
LABEL_25:
    if ( (_BYTE)v8 == 13 )
      return sub_ACADE0(*((_QWORD *)v34 + 1));
    if ( (unsigned int)(v8 - 12) <= 1 )
    {
      if ( (unsigned int)*v34 - 12 <= 1 )
        return (__int64)v34;
      return (__int64)a3;
    }
    if ( v34 == a3 )
      return (__int64)a3;
    v25 = *v34;
    if ( (_BYTE)v25 == 13 )
      return (__int64)a3;
    v26 = *a3;
    if ( (_BYTE)v26 == 13 )
      return (__int64)v34;
    if ( (unsigned int)(v25 - 12) > 1 )
      goto LABEL_38;
    if ( (_BYTE)v26 == 5 )
      return 0;
    if ( (unsigned __int8)v26 <= 0x14u )
    {
      v29 = 1441801;
      if ( _bittest64(&v29, v26) )
        return (__int64)a3;
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a3 + 1) + 8LL) - 17 > 1 )
        goto LABEL_38;
    }
    else if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)a3 + 1) + 8LL) - 17 > 1 )
    {
      return 0;
    }
    if ( !(unsigned __int8)sub_AD6C60(a3, a2, v26, v7) && !(unsigned __int8)sub_AD6CA0(a3) )
      return (__int64)a3;
    LOBYTE(v26) = *a3;
LABEL_38:
    if ( (unsigned int)(unsigned __int8)v26 - 12 <= 1 )
    {
      v27 = *v34;
      v28 = *v34 & 0xF7;
      if ( (*v34 & 0xF7) != 5 )
      {
        if ( (unsigned __int8)v27 <= 0x14u )
        {
          v28 = 1441801;
          if ( _bittest64(&v28, v27) )
            return (__int64)v34;
        }
        if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v34 + 1) + 8LL) - 17 <= 1
          && !(unsigned __int8)sub_AD6C60(v34, a2, v28, v7)
          && !(unsigned __int8)sub_AD6CA0(v34) )
        {
          return (__int64)v34;
        }
      }
    }
    return 0;
  }
  v9 = *((_QWORD *)a1 + 1);
  v35 = v37;
  v30 = v9;
  v36 = 0x1000000000LL;
  v10 = sub_BD5C60(a1, a2, v6);
  a2 = 32;
  v11 = sub_BCCE00(v10, 32);
  v12 = *(_DWORD *)(v9 + 32);
  v13 = v11;
  if ( v12 )
  {
    v14 = v12;
    v15 = 0;
    v33 = v14;
    while ( 1 )
    {
      v20 = sub_AD64C0(v13, v15, 0);
      v21 = (unsigned __int8 *)sub_AD5840(v34, v20, 0);
      a2 = sub_AD64C0(v13, v15, 0);
      v16 = sub_AD5840(a3, a2, 0);
      v22 = *(unsigned __int8 **)&a1[32 * (v15 - (*((_DWORD *)a1 + 1) & 0x7FFFFFF))];
      v23 = *v22;
      if ( *v22 == 13 )
      {
        v16 = sub_ACADE0(*((_QWORD *)v21 + 1));
      }
      else if ( v21 != (unsigned __int8 *)v16 )
      {
        if ( (unsigned int)v23 - 12 <= 1 )
        {
          if ( (unsigned int)*v21 - 12 <= 1 )
            v16 = (__int64)v21;
        }
        else
        {
          if ( v23 != 17 )
          {
            v12 = *(_DWORD *)(v30 + 32);
            v19 = v36;
            goto LABEL_21;
          }
          v31 = v16;
          v24 = sub_AC30F0(v22);
          v16 = v31;
          if ( !v24 )
            v16 = (__int64)v21;
        }
      }
      v17 = (unsigned int)v36;
      v18 = (unsigned int)v36 + 1LL;
      if ( v18 > HIDWORD(v36) )
      {
        a2 = (__int64)v37;
        v32 = v16;
        sub_C8D5F0(&v35, v37, v18, 8);
        v17 = (unsigned int)v36;
        v16 = v32;
      }
      ++v15;
      *(_QWORD *)&v35[8 * v17] = v16;
      v19 = v36 + 1;
      LODWORD(v36) = v36 + 1;
      if ( v15 == v33 )
      {
        v7 = v30;
        v12 = *(_DWORD *)(v30 + 32);
        goto LABEL_21;
      }
    }
  }
  v19 = v36;
LABEL_21:
  if ( v12 != v19 )
  {
    if ( v35 != v37 )
      _libc_free(v35, a2);
    v8 = *a1;
    goto LABEL_25;
  }
  result = sub_AD3730(v35, v12);
  if ( v35 != v37 )
  {
    v34 = (unsigned __int8 *)result;
    _libc_free(v35, v12);
    return (__int64)v34;
  }
  return result;
}

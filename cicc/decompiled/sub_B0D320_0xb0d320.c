// Function: sub_B0D320
// Address: 0xb0d320
//
__int64 __fastcall sub_B0D320(_QWORD *a1)
{
  unsigned __int64 *v1; // rbx
  unsigned __int64 *v2; // r13
  __int64 result; // rax
  unsigned __int64 *v4; // rax
  __int64 v5; // rbx
  unsigned __int64 *v6; // rax
  unsigned __int64 v7; // rdx
  _BYTE *v8; // r14
  __int64 v9; // rdx
  _BYTE *v10; // rax
  size_t v11; // r8
  __int64 v12; // rbx
  __int64 *v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // rdi
  unsigned __int64 v17; // rdx
  size_t v18; // [rsp+0h] [rbp-80h]
  __int64 v19; // [rsp+8h] [rbp-78h]
  _BYTE *v20; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v21; // [rsp+10h] [rbp-70h] BYREF
  __int64 v22; // [rsp+18h] [rbp-68h]
  _BYTE v23[96]; // [rsp+20h] [rbp-60h] BYREF

  v1 = (unsigned __int64 *)a1[2];
  v2 = (unsigned __int64 *)a1[3];
  v21 = v1;
  if ( v2 == v1 )
  {
    v4 = v1;
  }
  else
  {
    while ( *v1 != 4101 )
    {
      v1 += (unsigned int)sub_AF4160(&v21);
      v21 = v1;
      if ( v2 == v1 )
      {
        v1 = (unsigned __int64 *)a1[3];
        v4 = (unsigned __int64 *)a1[2];
        goto LABEL_8;
      }
    }
    result = (__int64)a1;
    if ( v2 != v1 )
      return result;
    v1 = (unsigned __int64 *)a1[3];
    v4 = (unsigned __int64 *)a1[2];
  }
LABEL_8:
  v5 = v1 - v4;
  v21 = (unsigned __int64 *)v23;
  v6 = (unsigned __int64 *)v23;
  v7 = (unsigned int)(v5 + 2);
  v22 = 0x600000000LL;
  if ( v7 > 6 )
  {
    sub_C8D5F0(&v21, v23, v7, 8);
    v17 = (unsigned int)v22 + 2LL;
    if ( v17 > HIDWORD(v22) )
      sub_C8D5F0(&v21, v23, v17, 8);
    v6 = &v21[(unsigned int)v22];
  }
  *v6 = 4101;
  v6[1] = 0;
  v8 = (_BYTE *)a1[2];
  v10 = (_BYTE *)a1[3];
  LODWORD(v22) = v22 + 2;
  v9 = (unsigned int)v22;
  v11 = v10 - v8;
  v12 = (v10 - v8) >> 3;
  if ( v12 + (unsigned __int64)(unsigned int)v22 > HIDWORD(v22) )
  {
    v18 = v10 - v8;
    v20 = v10;
    sub_C8D5F0(&v21, v23, v12 + (unsigned int)v22, 8);
    v9 = (unsigned int)v22;
    v11 = v18;
    v10 = v20;
  }
  v13 = (__int64 *)v21;
  if ( v8 != v10 )
  {
    memcpy(&v21[v9], v8, v11);
    v13 = (__int64 *)v21;
    LODWORD(v9) = v22;
  }
  v14 = a1[1];
  LODWORD(v22) = v12 + v9;
  v15 = (unsigned int)(v12 + v9);
  v16 = (__int64 *)(v14 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v14 & 4) != 0 )
    v16 = (__int64 *)*v16;
  result = sub_B0D000(v16, v13, v15, 0, 1);
  if ( v21 != (unsigned __int64 *)v23 )
  {
    v19 = result;
    _libc_free(v21, v13);
    return v19;
  }
  return result;
}

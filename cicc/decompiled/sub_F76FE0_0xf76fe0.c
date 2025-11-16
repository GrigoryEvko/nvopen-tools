// Function: sub_F76FE0
// Address: 0xf76fe0
//
signed __int64 __fastcall sub_F76FE0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  signed __int64 result; // rax
  __int64 v8; // r8
  _BYTE *v9; // rdx
  __int64 v10; // rax
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  _BYTE *v16; // rcx
  __int64 v17; // r14
  _BYTE *v18; // rbx
  _BYTE *v19; // r15
  __int64 v20; // r13
  unsigned __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-D0h]
  signed __int64 v23; // [rsp+10h] [rbp-C0h]
  __int64 v24; // [rsp+28h] [rbp-A8h]
  _BYTE *v25; // [rsp+40h] [rbp-90h] BYREF
  __int64 v26; // [rsp+48h] [rbp-88h]
  _BYTE v27[32]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v28; // [rsp+70h] [rbp-60h] BYREF
  __int64 v29; // [rsp+78h] [rbp-58h]
  _BYTE v30[80]; // [rsp+80h] [rbp-50h] BYREF

  v25 = v27;
  v26 = 0x400000000LL;
  v28 = v30;
  v6 = *a1;
  v29 = 0x400000000LL;
  result = a1[1];
  v23 = result;
  if ( result != v6 )
  {
    v8 = *(_QWORD *)(v6 - 8);
    v9 = v30;
    v24 = v6 - 8;
    v10 = 0;
    while ( 1 )
    {
      *(_QWORD *)&v9[8 * v10] = v8;
      v11 = v29 + 1;
      LODWORD(v29) = v29 + 1;
      do
      {
        v16 = v28;
        v17 = *(_QWORD *)&v28[8 * v11 - 8];
        v12 = (unsigned int)(v11 - 1);
        LODWORD(v29) = v11 - 1;
        v18 = *(_BYTE **)(v17 + 16);
        v19 = *(_BYTE **)(v17 + 8);
        v20 = (v18 - v19) >> 3;
        if ( v20 + v12 > (unsigned __int64)HIDWORD(v29) )
        {
          sub_C8D5F0((__int64)&v28, v30, v20 + v12, 8u, v8, a6);
          v16 = v28;
          v12 = (unsigned int)v29;
        }
        if ( v19 != v18 )
        {
          memmove(&v16[8 * v12], v19, v18 - v19);
          LODWORD(v12) = v29;
        }
        v13 = HIDWORD(v26);
        LODWORD(v29) = v20 + v12;
        v14 = (unsigned int)v26;
        v15 = (unsigned int)v26 + 1LL;
        if ( v15 > HIDWORD(v26) )
        {
          sub_C8D5F0((__int64)&v25, v27, v15, 8u, v8, a6);
          v14 = (unsigned int)v26;
        }
        *(_QWORD *)&v25[8 * v14] = v17;
        v11 = v29;
        LODWORD(v26) = v26 + 1;
      }
      while ( (_DWORD)v29 );
      result = sub_F769A0(a2, (__int64)&v25, 0, v13, v8, a6);
      LODWORD(v26) = 0;
      if ( v23 == v24 )
        break;
      v8 = *(_QWORD *)(v24 - 8);
      v10 = (unsigned int)v29;
      v21 = (unsigned int)v29 + 1LL;
      if ( v21 > HIDWORD(v29) )
      {
        v22 = *(_QWORD *)(v24 - 8);
        sub_C8D5F0((__int64)&v28, v30, v21, 8u, v8, a6);
        v10 = (unsigned int)v29;
        v8 = v22;
      }
      v24 -= 8;
      v9 = v28;
    }
    if ( v28 != v30 )
      result = _libc_free(v28, &v25);
    if ( v25 != v27 )
      return _libc_free(v25, &v25);
  }
  return result;
}

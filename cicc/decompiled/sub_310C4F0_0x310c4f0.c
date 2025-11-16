// Function: sub_310C4F0
// Address: 0x310c4f0
//
void __fastcall sub_310C4F0(__int64 *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // r13
  __int64 *v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // r8
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rdi
  _QWORD *v14; // rax
  __int64 *v15; // rdi
  _QWORD *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-A8h]
  __int64 v18; // [rsp+8h] [rbp-A8h]
  __int64 *v19; // [rsp+28h] [rbp-88h]
  __int64 v20; // [rsp+30h] [rbp-80h] BYREF
  __int64 v21; // [rsp+38h] [rbp-78h] BYREF
  _BYTE *v22; // [rsp+40h] [rbp-70h] BYREF
  __int64 v23; // [rsp+48h] [rbp-68h]
  _BYTE v24[16]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v25; // [rsp+60h] [rbp-50h] BYREF
  __int64 v26; // [rsp+68h] [rbp-48h]
  _BYTE v27[64]; // [rsp+70h] [rbp-40h] BYREF

  v3 = a1[1];
  v22 = v24;
  v25 = v27;
  v23 = 0x200000000LL;
  v26 = 0x200000000LL;
  v4 = sub_D95540(v3);
  v5 = *(__int64 **)(a2 + 32);
  v19 = &v5[*(_QWORD *)(a2 + 40)];
  if ( v19 != v5 )
  {
    while ( 1 )
    {
      sub_310BF50(*a1, *v5, a1[1], &v20, &v21);
      if ( v4 != sub_D95540(v20) || v4 != sub_D95540(v21) )
        break;
      v10 = (unsigned int)v23;
      v11 = v20;
      v12 = (unsigned int)v23 + 1LL;
      if ( v12 > HIDWORD(v23) )
      {
        v17 = v20;
        sub_C8D5F0((__int64)&v22, v24, v12, 8u, v20, v9);
        v10 = (unsigned int)v23;
        v11 = v17;
      }
      *(_QWORD *)&v22[8 * v10] = v11;
      v6 = (unsigned int)v26;
      LODWORD(v23) = v23 + 1;
      v7 = (unsigned int)v26 + 1LL;
      v8 = v21;
      if ( v7 > HIDWORD(v26) )
      {
        v18 = v21;
        sub_C8D5F0((__int64)&v25, v27, v7, 8u, v21, v9);
        v6 = (unsigned int)v26;
        v8 = v18;
      }
      ++v5;
      *(_QWORD *)&v25[8 * v6] = v8;
      LODWORD(v26) = v26 + 1;
      if ( v19 == v5 )
        goto LABEL_16;
    }
    sub_310A840(a1, a2);
    v13 = v25;
LABEL_11:
    if ( v13 == v27 )
      goto LABEL_13;
    goto LABEL_12;
  }
LABEL_16:
  if ( (_DWORD)v23 == 1 )
  {
    v13 = v25;
    a1[2] = *(_QWORD *)v22;
    a1[3] = *(_QWORD *)v13;
    goto LABEL_11;
  }
  v14 = sub_DC7EB0((__int64 *)*a1, (__int64)&v22, 0, 0);
  v15 = (__int64 *)*a1;
  a1[2] = (__int64)v14;
  v16 = sub_DC7EB0(v15, (__int64)&v25, 0, 0);
  v13 = v25;
  a1[3] = (__int64)v16;
  if ( v13 != v27 )
LABEL_12:
    _libc_free((unsigned __int64)v13);
LABEL_13:
  if ( v22 != v24 )
    _libc_free((unsigned __int64)v22);
}

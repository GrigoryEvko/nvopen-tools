// Function: sub_B0E7F0
// Address: 0xb0e7f0
//
__int64 __fastcall sub_B0E7F0(_QWORD *a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // r15
  unsigned __int64 *v3; // r12
  char v4; // r14
  char v5; // bl
  unsigned __int64 *v6; // r9
  unsigned __int64 *v7; // r15
  __int64 v8; // rax
  size_t v9; // r10
  __int64 v10; // rbx
  unsigned __int64 v11; // rdx
  unsigned __int64 *v12; // r15
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  __int64 *v15; // rdi
  __int64 *v16; // r9
  __int64 v17; // r12
  __int64 *v18; // rdi
  unsigned __int64 *v20; // [rsp+8h] [rbp-D8h]
  size_t v22; // [rsp+20h] [rbp-C0h]
  unsigned __int64 *v23; // [rsp+28h] [rbp-B8h]
  void *src; // [rsp+30h] [rbp-B0h] BYREF
  unsigned __int64 *v25; // [rsp+38h] [rbp-A8h] BYREF
  unsigned __int64 v26; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v27; // [rsp+48h] [rbp-98h]
  unsigned __int64 v28; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v29; // [rsp+58h] [rbp-88h]
  __int64 *v30; // [rsp+60h] [rbp-80h] BYREF
  __int64 v31; // [rsp+68h] [rbp-78h]
  _BYTE v32[112]; // [rsp+70h] [rbp-70h] BYREF

  v20 = a2;
  v27 = *((_DWORD *)a2 + 8);
  if ( v27 > 0x40 )
  {
    a2 += 3;
    sub_C43780(&v26, v20 + 3);
  }
  else
  {
    v26 = a2[3];
  }
  v30 = (__int64 *)v32;
  v31 = 0x800000000LL;
  v2 = (unsigned __int64 *)a1[2];
  v3 = (unsigned __int64 *)a1[3];
  v25 = v2;
  if ( v3 != v2 )
  {
    v4 = 0;
    v5 = 1;
    while ( 1 )
    {
      src = v25;
      if ( *v2 != 4097 )
      {
        if ( !v4 )
        {
          v17 = (__int64)a1;
          v15 = v30;
          goto LABEL_27;
        }
LABEL_6:
        v6 = &v2[(unsigned int)sub_AF4160((unsigned __int64 **)&src)];
        v7 = (unsigned __int64 *)src;
        v8 = (unsigned int)v31;
        v9 = (char *)v6 - (_BYTE *)src;
        v10 = ((char *)v6 - (_BYTE *)src) >> 3;
        v11 = v10 + (unsigned int)v31;
        if ( v11 > HIDWORD(v31) )
        {
          a2 = (unsigned __int64 *)v32;
          v22 = (char *)v6 - (_BYTE *)src;
          v23 = v6;
          sub_C8D5F0(&v30, v32, v11, 8);
          v8 = (unsigned int)v31;
          v9 = v22;
          v6 = v23;
        }
        if ( v6 != v7 )
        {
          a2 = v7;
          memcpy(&v30[v8], v7, v9);
          LODWORD(v8) = v31;
        }
        LODWORD(v31) = v8 + v10;
        v5 = 0;
        goto LABEL_11;
      }
      if ( !v5 )
        goto LABEL_6;
      v13 = v2[1];
      a2 = &v26;
      if ( v2[2] == 5 )
      {
        sub_C44B10(&v28, &v26, v13);
        if ( v27 <= 0x40 )
          goto LABEL_18;
        v14 = v26;
        if ( !v26 )
          goto LABEL_18;
      }
      else
      {
        sub_C44AB0(&v28, &v26, v13);
        if ( v27 <= 0x40 )
          goto LABEL_18;
        v14 = v26;
        if ( !v26 )
          goto LABEL_18;
      }
      j_j___libc_free_0_0(v14);
LABEL_18:
      v4 = v5;
      v26 = v28;
      v27 = v29;
LABEL_11:
      v12 = v25;
      v2 = &v12[(unsigned int)sub_AF4160(&v25)];
      v25 = v2;
      if ( v3 == v2 )
      {
        v15 = v30;
        if ( !v4 )
          goto LABEL_26;
        v16 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
        if ( (a1[1] & 4) != 0 )
          v16 = (__int64 *)*v16;
        v17 = sub_B0D000(v16, v30, (unsigned int)v31, 0, 1);
        v18 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
        if ( (a1[1] & 4) != 0 )
          v18 = (__int64 *)*v18;
        a2 = &v26;
        sub_ACCFD0(v18, (__int64)&v26);
        v15 = v30;
        goto LABEL_27;
      }
    }
  }
  v15 = (__int64 *)v32;
LABEL_26:
  v17 = (__int64)a1;
LABEL_27:
  if ( v15 != (__int64 *)v32 )
    _libc_free(v15, a2);
  if ( v27 > 0x40 && v26 )
    j_j___libc_free_0_0(v26);
  return v17;
}

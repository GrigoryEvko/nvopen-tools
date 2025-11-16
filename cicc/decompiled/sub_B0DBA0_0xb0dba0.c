// Function: sub_B0DBA0
// Address: 0xb0dba0
//
__int64 __fastcall sub_B0DBA0(_QWORD *a1, _BYTE *a2, __int64 a3, int a4, char a5)
{
  unsigned __int64 *v5; // r14
  unsigned __int64 *v6; // rbx
  char v7; // r13
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // rax
  char *v12; // r15
  int v13; // eax
  char *v14; // r9
  char *v15; // r10
  __int64 v16; // rax
  size_t v17; // r11
  __int64 v18; // r15
  unsigned __int64 v19; // rdx
  unsigned __int64 **v20; // rsi
  __int64 *v21; // rdi
  __int64 v22; // rax
  unsigned __int64 *v23; // rdi
  __int64 v24; // r12
  __int64 v26; // rax
  _BYTE *v27; // [rsp+0h] [rbp-D0h]
  size_t v30; // [rsp+28h] [rbp-A8h]
  char *v32; // [rsp+30h] [rbp-A0h]
  char *v34; // [rsp+38h] [rbp-98h]
  void *src; // [rsp+40h] [rbp-90h] BYREF
  unsigned __int64 *v36; // [rsp+48h] [rbp-88h] BYREF
  unsigned __int64 *v37; // [rsp+50h] [rbp-80h] BYREF
  __int64 v38; // [rsp+58h] [rbp-78h]
  _BYTE v39[112]; // [rsp+60h] [rbp-70h] BYREF

  v5 = (unsigned __int64 *)a1[2];
  v6 = (unsigned __int64 *)a1[3];
  v37 = v5;
  if ( v5 == v6 )
    goto LABEL_28;
  v7 = a5;
  while ( *v5 != 4101 )
  {
    v5 += (unsigned int)sub_AF4160(&v37);
    v37 = v5;
    if ( v6 == v5 )
      goto LABEL_28;
  }
  if ( v6 == v5 )
  {
LABEL_28:
    v38 = 0x800000000LL;
    v37 = (unsigned __int64 *)v39;
    sub_AF5AE0((__int64)&v37, a2, &a2[8 * a3]);
    v20 = &v37;
    v26 = sub_B0D8A0(a1, (__int64)&v37, a5, 0);
    v23 = v37;
    v24 = v26;
    if ( v37 == (unsigned __int64 *)v39 )
      return v24;
    goto LABEL_26;
  }
  v37 = (unsigned __int64 *)v39;
  v8 = (unsigned __int64 *)a1[3];
  v38 = 0x800000000LL;
  v9 = (unsigned __int64 *)a1[2];
  v36 = v9;
  if ( v8 != v9 )
  {
    v27 = &a2[8 * a3];
    do
    {
      src = v36;
      if ( v7 )
      {
        v11 = *v9;
        if ( v11 == 159 )
        {
          v7 = 0;
        }
        else if ( v11 == 4096 )
        {
          v7 = 0;
          sub_A188E0((__int64)&v37, 159);
        }
      }
      v12 = (char *)src;
      v13 = sub_AF4160((unsigned __int64 **)&src);
      v14 = (char *)src;
      v15 = &v12[8 * v13];
      v16 = (unsigned int)v38;
      v17 = v15 - (_BYTE *)src;
      v18 = (v15 - (_BYTE *)src) >> 3;
      v19 = v18 + (unsigned int)v38;
      if ( v19 > HIDWORD(v38) )
      {
        v30 = v15 - (_BYTE *)src;
        v32 = (char *)src;
        v34 = v15;
        sub_C8D5F0(&v37, v39, v19, 8);
        v16 = (unsigned int)v38;
        v17 = v30;
        v14 = v32;
        v15 = v34;
      }
      if ( v15 != v14 )
      {
        memcpy(&v37[v16], v14, v17);
        LODWORD(v16) = v38;
      }
      LODWORD(v38) = v18 + v16;
      if ( *(_QWORD *)src == 4101 && a4 == *((_QWORD *)src + 1) )
        sub_AF5AE0((__int64)&v37, a2, v27);
      v10 = v36;
      v9 = &v10[(unsigned int)sub_AF4160(&v36)];
      v36 = v9;
    }
    while ( v8 != v9 );
  }
  if ( v7 )
    sub_A188E0((__int64)&v37, 159);
  v20 = (unsigned __int64 **)v37;
  v21 = (__int64 *)(a1[1] & 0xFFFFFFFFFFFFFFF8LL);
  if ( (a1[1] & 4) != 0 )
    v21 = (__int64 *)*v21;
  v22 = sub_B0D000(v21, (__int64 *)v37, (unsigned int)v38, 0, 1);
  v23 = v37;
  v24 = v22;
  if ( v37 != (unsigned __int64 *)v39 )
LABEL_26:
    _libc_free(v23, v20);
  return v24;
}

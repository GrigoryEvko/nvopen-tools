// Function: sub_E89A10
// Address: 0xe89a10
//
unsigned __int64 __fastcall sub_E89A10(__int64 a1, _BYTE *a2, unsigned __int64 a3)
{
  __int64 v3; // r12
  unsigned __int64 v4; // rcx
  char *v5; // r13
  size_t v6; // r15
  _QWORD *v7; // rax
  bool v8; // zf
  __int64 v9; // rax
  _QWORD *v10; // rdi
  unsigned __int64 v11; // r12
  char *v13; // r13
  size_t v14; // r15
  _QWORD *v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rax
  _BYTE *v18; // rdi
  _QWORD *v19; // rdi
  unsigned __int64 v20; // rax
  _BYTE *v21; // rdi
  _QWORD *v22; // rdi
  _QWORD *v23; // [rsp+10h] [rbp-B0h] BYREF
  size_t v24; // [rsp+18h] [rbp-A8h]
  _QWORD v25[2]; // [rsp+20h] [rbp-A0h] BYREF
  void *v26[4]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v27; // [rsp+50h] [rbp-70h]
  _QWORD v28[2]; // [rsp+60h] [rbp-60h] BYREF
  char v29; // [rsp+74h] [rbp-4Ch] BYREF
  _BYTE v30[11]; // [rsp+75h] [rbp-4Bh] BYREF
  __int16 v31; // [rsp+80h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 920);
  v4 = a3;
  switch ( *(_DWORD *)(v3 + 76) )
  {
    case 0:
    case 1:
    case 2:
    case 4:
    case 5:
    case 6:
    case 8:
      sub_C64ED0("Cannot get DWARF comdat section for this object file format: not implemented.", 1u);
    case 3:
      if ( !a3 )
      {
        v29 = 48;
        v13 = &v29;
        v23 = v25;
LABEL_12:
        v14 = 1;
        LOBYTE(v25[0]) = *v13;
        v15 = v25;
        goto LABEL_13;
      }
      v13 = v30;
      do
      {
        *--v13 = v4 % 0xA + 48;
        v17 = v4;
        v4 /= 0xAu;
      }
      while ( v17 > 9 );
      v18 = (_BYTE *)(v30 - v13);
      v23 = v25;
      v14 = v30 - v13;
      v26[0] = (void *)(v30 - v13);
      if ( (unsigned __int64)(v30 - v13) > 0xF )
      {
        v23 = (_QWORD *)sub_22409D0(&v23, v26, 0);
        v19 = v23;
        v25[0] = v26[0];
LABEL_21:
        memcpy(v19, v13, v14);
        v14 = (size_t)v26[0];
        v15 = v23;
        goto LABEL_13;
      }
      if ( v18 == (_BYTE *)1 )
        goto LABEL_12;
      if ( v18 )
      {
        v19 = v25;
        goto LABEL_21;
      }
      v15 = v25;
LABEL_13:
      v24 = v14;
      *((_BYTE *)v15 + v14) = 0;
      v8 = *a2 == 0;
      v31 = 260;
      v28[0] = &v23;
      v27 = 257;
      if ( !v8 )
      {
        v26[0] = a2;
        LOBYTE(v27) = 3;
      }
      v16 = sub_E71CB0(v3, (size_t *)v26, 1, 0x200u, 0, (__int64)v28, 1u, -1, 0);
      v10 = v23;
      v11 = v16;
      if ( v23 == v25 )
        return v11;
      goto LABEL_8;
    case 7:
      if ( !a3 )
      {
        v29 = 48;
        v5 = &v29;
        v23 = v25;
LABEL_4:
        v6 = 1;
        LOBYTE(v25[0]) = *v5;
        v7 = v25;
        goto LABEL_5;
      }
      v5 = v30;
      do
      {
        *--v5 = v4 % 0xA + 48;
        v20 = v4;
        v4 /= 0xAu;
      }
      while ( v20 > 9 );
      v21 = (_BYTE *)(v30 - v5);
      v23 = v25;
      v6 = v30 - v5;
      v26[0] = (void *)(v30 - v5);
      if ( (unsigned __int64)(v30 - v5) > 0xF )
      {
        v23 = (_QWORD *)sub_22409D0(&v23, v26, 0);
        v22 = v23;
        v25[0] = v26[0];
LABEL_26:
        memcpy(v22, v5, v6);
        v6 = (size_t)v26[0];
        v7 = v23;
        goto LABEL_5;
      }
      if ( v21 == (_BYTE *)1 )
        goto LABEL_4;
      if ( v21 )
      {
        v22 = v25;
        goto LABEL_26;
      }
      v7 = v25;
LABEL_5:
      v24 = v6;
      *((_BYTE *)v7 + v6) = 0;
      v8 = *a2 == 0;
      v31 = 260;
      v28[0] = &v23;
      v27 = 257;
      if ( !v8 )
      {
        v26[0] = a2;
        LOBYTE(v27) = 3;
      }
      v9 = sub_E6D8A0(v3, v26, 0, 0, (__int64)v28, -1);
      v10 = v23;
      v11 = v9;
      if ( v23 != v25 )
LABEL_8:
        j_j___libc_free_0(v10, v25[0] + 1LL);
      return v11;
    default:
      BUG();
  }
}

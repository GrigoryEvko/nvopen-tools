// Function: sub_EAE180
// Address: 0xeae180
//
__int64 __fastcall sub_EAE180(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  unsigned int v4; // edx
  char *v5; // r14
  char v6; // al
  char *v7; // r12
  _BYTE *v9; // r14
  _QWORD *v10; // r9
  size_t v11; // r12
  _QWORD *v12; // rdx
  size_t v13; // rsi
  size_t v14; // r13
  char v15; // al
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // r15
  _BYTE *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // [rsp+8h] [rbp-68h]
  char v22; // [rsp+17h] [rbp-59h]
  _QWORD *v23; // [rsp+18h] [rbp-58h]
  _QWORD *v24; // [rsp+20h] [rbp-50h] BYREF
  size_t n; // [rsp+28h] [rbp-48h]
  _QWORD src[8]; // [rsp+30h] [rbp-40h] BYREF

  v3 = sub_ECD7B0(a1);
  v5 = (char *)sub_ECD6A0(v3);
  v6 = *v5;
  v7 = v5;
  if ( *v5 == 10 || v6 == 62 )
  {
LABEL_8:
    v4 = 1;
    if ( v6 != 62 )
      return v4;
    v9 = v5 + 1;
    sub_EA24B0(a1, (unsigned __int64)(v7 + 1), *(_DWORD *)(a1 + 304));
    sub_EABFE0(a1);
    v10 = src;
    LOBYTE(src[0]) = 0;
    v24 = src;
    n = 0;
    v11 = v7 - v9;
    if ( v11 )
    {
      v12 = src;
      v13 = 0;
      v14 = 0;
      while ( 1 )
      {
        v15 = v9[v14];
        if ( v15 == 33 )
          v15 = v9[++v14];
        v16 = 15;
        v17 = v13 + 1;
        if ( v12 != v10 )
          v16 = src[0];
        if ( v17 > v16 )
        {
          v21 = v10;
          v22 = v15;
          sub_2240BB0(&v24, v13, 0, 0, 1);
          v12 = v24;
          v10 = v21;
          v15 = v22;
        }
        *((_BYTE *)v12 + v13) = v15;
        ++v14;
        n = v13 + 1;
        *((_BYTE *)v24 + v17) = 0;
        if ( v11 <= v14 )
          break;
        v13 = n;
        v12 = v24;
      }
      v11 = n;
      v18 = (_BYTE *)*a2;
      if ( v24 != v10 )
      {
        v19 = src[0];
        if ( v18 == (_BYTE *)(a2 + 2) )
        {
          *a2 = v24;
          a2[1] = v11;
          a2[2] = v19;
        }
        else
        {
          v20 = a2[2];
          *a2 = v24;
          a2[1] = v11;
          a2[2] = v19;
          if ( v18 )
          {
            v24 = v18;
            src[0] = v20;
LABEL_23:
            n = 0;
            *v18 = 0;
            if ( v24 != v10 )
              j_j___libc_free_0(v24, src[0] + 1LL);
            return 0;
          }
        }
        v24 = v10;
        v10 = src;
        v18 = src;
        goto LABEL_23;
      }
      if ( !n )
      {
LABEL_29:
        a2[1] = v11;
        v18[v11] = 0;
        v18 = v24;
        goto LABEL_23;
      }
      if ( n != 1 )
      {
        v23 = v10;
        memcpy(v18, v10, n);
        v11 = n;
        v18 = (_BYTE *)*a2;
        v10 = v23;
        goto LABEL_29;
      }
      *v18 = src[0];
      v11 = n;
    }
    v18 = (_BYTE *)*a2;
    goto LABEL_29;
  }
  while ( 1 )
  {
    LOBYTE(v4) = v6 == 0 || v6 == 13;
    if ( (_BYTE)v4 )
      return v4;
    v7 += (v6 == 33) + 1;
    v6 = *v7;
    if ( *v7 == 62 || v6 == 10 )
      goto LABEL_8;
  }
}

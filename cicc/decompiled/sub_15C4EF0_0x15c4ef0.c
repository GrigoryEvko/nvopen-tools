// Function: sub_15C4EF0
// Address: 0x15c4ef0
//
__int64 __fastcall sub_15C4EF0(__int64 a1, _QWORD *a2, unsigned int a3, unsigned int a4)
{
  unsigned __int64 *v5; // rdx
  unsigned __int64 *v6; // r15
  unsigned __int64 *v7; // r13
  unsigned __int64 v8; // rax
  int v9; // eax
  void *v10; // r9
  __int64 v11; // rdx
  unsigned __int64 *v12; // rax
  size_t v13; // r10
  unsigned __int64 v14; // r13
  unsigned __int64 *v15; // r13
  __int64 v16; // rax
  char *v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 *v22; // rdi
  __int64 v23; // rax
  size_t v25; // [rsp+8h] [rbp-B8h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-B0h]
  unsigned __int64 *v27; // [rsp+18h] [rbp-A8h]
  void *src; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 *v31; // [rsp+38h] [rbp-88h] BYREF
  char *v32; // [rsp+40h] [rbp-80h] BYREF
  __int64 v33; // [rsp+48h] [rbp-78h]
  _BYTE v34[112]; // [rsp+50h] [rbp-70h] BYREF

  v32 = v34;
  v33 = 0x800000000LL;
  if ( a2 )
  {
    v5 = (unsigned __int64 *)a2[3];
    v6 = (unsigned __int64 *)a2[4];
    v31 = v5;
    if ( v6 == v5 )
    {
      v16 = 0;
LABEL_13:
      v17 = &v32[8 * v16];
      goto LABEL_14;
    }
    v7 = v5;
    while ( 1 )
    {
      src = v31;
      v8 = *v7;
      if ( *v7 == 34 )
      {
LABEL_22:
        *(_BYTE *)(a1 + 8) = 0;
        goto LABEL_23;
      }
      if ( v8 == 4096 )
      {
        a3 += *((_DWORD *)v7 + 2);
        goto LABEL_11;
      }
      if ( v8 == 28 )
        goto LABEL_22;
      v9 = sub_15B11B0((unsigned __int64 **)&src);
      v10 = src;
      v11 = (unsigned int)v33;
      v12 = &v7[v9];
      v13 = (char *)v12 - (_BYTE *)src;
      v14 = ((char *)v12 - (_BYTE *)src) >> 3;
      if ( v14 > HIDWORD(v33) - (unsigned __int64)(unsigned int)v33 )
        break;
      if ( v12 != src )
        goto LABEL_9;
LABEL_10:
      LODWORD(v33) = v14 + v11;
LABEL_11:
      v15 = v31;
      v7 = &v15[(unsigned int)sub_15B11B0(&v31)];
      v31 = v7;
      if ( v6 == v7 )
      {
        v16 = (unsigned int)v33;
        if ( (unsigned int)v33 < HIDWORD(v33) )
          goto LABEL_13;
        sub_16CD150(&v32, v34, 0, 8);
        v17 = &v32[8 * (unsigned int)v33];
        goto LABEL_14;
      }
    }
    v25 = (char *)v12 - (_BYTE *)src;
    v26 = (unsigned __int64 *)src;
    v27 = v12;
    sub_16CD150(&v32, v34, v14 + (unsigned int)v33, 8);
    v10 = v26;
    v11 = (unsigned int)v33;
    v13 = v25;
    if ( v27 == v26 )
      goto LABEL_10;
LABEL_9:
    memcpy(&v32[8 * v11], v10, v13);
    LODWORD(v11) = v33;
    goto LABEL_10;
  }
  v17 = v34;
LABEL_14:
  *(_QWORD *)v17 = 4096;
  v18 = (unsigned int)(v33 + 1);
  LODWORD(v33) = v18;
  if ( HIDWORD(v33) <= (unsigned int)v18 )
  {
    sub_16CD150(&v32, v34, 0, 8);
    v18 = (unsigned int)v33;
  }
  *(_QWORD *)&v32[8 * v18] = a3;
  v19 = (unsigned int)(v33 + 1);
  LODWORD(v33) = v19;
  if ( HIDWORD(v33) <= (unsigned int)v19 )
  {
    sub_16CD150(&v32, v34, 0, 8);
    v19 = (unsigned int)v33;
  }
  *(_QWORD *)&v32[8 * v19] = a4;
  v20 = (unsigned int)(v33 + 1);
  v21 = a2[2];
  LODWORD(v33) = v33 + 1;
  v22 = (__int64 *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v21 & 4) != 0 )
    v22 = (__int64 *)*v22;
  v23 = sub_15C4420(v22, v32, v20, 0, 1);
  *(_BYTE *)(a1 + 8) = 1;
  *(_QWORD *)a1 = v23;
LABEL_23:
  if ( v32 != v34 )
    _libc_free((unsigned __int64)v32);
  return a1;
}

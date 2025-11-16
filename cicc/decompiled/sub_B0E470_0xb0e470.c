// Function: sub_B0E470
// Address: 0xb0e470
//
__int64 __fastcall sub_B0E470(__int64 a1, unsigned int a2, unsigned int a3)
{
  __int64 v3; // rbx
  unsigned __int64 *v4; // r13
  unsigned __int64 *v5; // rax
  unsigned __int64 *v6; // rbx
  char v7; // r13
  __int64 *v8; // rsi
  unsigned __int64 v9; // rax
  __int64 v10; // rbx
  int v12; // eax
  unsigned __int64 *v13; // r12
  __int64 v14; // rdx
  unsigned __int64 *v15; // rax
  size_t v16; // r11
  __int64 v17; // rbx
  unsigned __int64 *v18; // rbx
  __int64 *v19; // rdi
  __int64 v20; // rax
  size_t v21; // [rsp+8h] [rbp-D8h]
  unsigned __int64 *v22; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v23; // [rsp+10h] [rbp-D0h]
  char v24; // [rsp+18h] [rbp-C8h]
  unsigned __int64 *v27; // [rsp+28h] [rbp-B8h]
  void *src; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 *v29; // [rsp+48h] [rbp-98h] BYREF
  __int64 v30; // [rsp+50h] [rbp-90h]
  __int64 v31; // [rsp+58h] [rbp-88h]
  __int64 *v32; // [rsp+60h] [rbp-80h] BYREF
  __int64 v33; // [rsp+68h] [rbp-78h]
  _BYTE v34[112]; // [rsp+70h] [rbp-70h] BYREF

  v3 = a1;
  v32 = (__int64 *)v34;
  v33 = 0x800000000LL;
  if ( !a1 || (v4 = *(unsigned __int64 **)(a1 + 16), v5 = *(unsigned __int64 **)(a1 + 24), v29 = v4, v27 = v5, v5 == v4) )
  {
LABEL_24:
    sub_A188E0((__int64)&v32, 4096);
    sub_A188E0((__int64)&v32, a2);
    sub_A188E0((__int64)&v32, a3);
    goto LABEL_25;
  }
  v24 = 1;
  v6 = v4;
  v7 = 1;
  do
  {
    src = v29;
    v8 = (__int64 *)*v6;
    if ( *v6 == 159 )
    {
      if ( !v7 )
        goto LABEL_11;
      goto LABEL_17;
    }
    if ( *v6 <= 0x9F )
    {
      if ( v8 == (__int64 *)28 )
      {
        v7 = 0;
      }
      else if ( (unsigned __int64)v8 <= 0x1C )
      {
        if ( v8 == (__int64 *)6 )
        {
          v7 = 1;
        }
        else if ( v8 == (__int64 *)24 )
        {
          v7 = 1;
        }
      }
      else if ( (unsigned __int64)v8 > 0x26 )
      {
        if ( (unsigned __int64)v8 - 148 <= 1 )
          v7 = 1;
      }
      else if ( (unsigned __int64)v8 >= 0x22 )
      {
        v7 = 0;
      }
      goto LABEL_17;
    }
    if ( v8 == (__int64 *)4096 )
    {
      if ( !v24 )
        goto LABEL_11;
      a2 += *((_DWORD *)v6 + 2);
      goto LABEL_22;
    }
    if ( (unsigned __int64)v8 <= 0x1000 )
    {
      if ( (unsigned __int64)v8 - 166 <= 1 )
        v7 = 1;
LABEL_17:
      v12 = sub_AF4160((unsigned __int64 **)&src);
      v13 = (unsigned __int64 *)src;
      v14 = (unsigned int)v33;
      v15 = &v6[v12];
      v16 = (char *)v15 - (_BYTE *)src;
      v17 = ((char *)v15 - (_BYTE *)src) >> 3;
      if ( v17 + (unsigned __int64)(unsigned int)v33 > HIDWORD(v33) )
      {
        v21 = (char *)v15 - (_BYTE *)src;
        v22 = v15;
        sub_C8D5F0(&v32, v34, v17 + (unsigned int)v33, 8);
        v14 = (unsigned int)v33;
        v16 = v21;
        v15 = v22;
      }
      if ( v15 != v13 )
      {
        memcpy(&v32[v14], v13, v16);
        LODWORD(v14) = v33;
      }
      LODWORD(v33) = v17 + v14;
      goto LABEL_22;
    }
    if ( (unsigned __int64)v8 - 4102 > 1 )
      goto LABEL_17;
    v9 = v6[1];
    v10 = v6[2];
    if ( a2 > v9 || v10 + v9 > a3 + a2 )
    {
LABEL_11:
      LOBYTE(v31) = 0;
      goto LABEL_12;
    }
    v23 = v9;
    sub_A188E0((__int64)&v32, (__int64)v8);
    sub_A188E0((__int64)&v32, v23 - a2);
    sub_A188E0((__int64)&v32, v10);
    v24 = 0;
LABEL_22:
    v18 = v29;
    v6 = &v18[(unsigned int)sub_AF4160(&v29)];
    v29 = v6;
  }
  while ( v27 != v6 );
  v3 = a1;
  if ( v24 )
    goto LABEL_24;
LABEL_25:
  v8 = v32;
  v19 = (__int64 *)(*(_QWORD *)(v3 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v3 + 8) & 4) != 0 )
    v19 = (__int64 *)*v19;
  v20 = sub_B0D000(v19, v32, (unsigned int)v33, 0, 1);
  LOBYTE(v31) = 1;
  v30 = v20;
LABEL_12:
  if ( v32 != (__int64 *)v34 )
    _libc_free(v32, v8);
  return v30;
}

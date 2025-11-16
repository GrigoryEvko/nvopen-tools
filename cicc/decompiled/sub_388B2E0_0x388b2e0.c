// Function: sub_388B2E0
// Address: 0x388b2e0
//
__int64 __fastcall sub_388B2E0(__int64 a1)
{
  __int64 v1; // r14
  int v2; // eax
  unsigned __int64 v3; // rsi
  unsigned int v4; // r12d
  __int8 *v6; // r8
  _QWORD *v7; // rbx
  unsigned __int64 v8; // r15
  const char *v9; // rax
  char *v10; // rdi
  const char *v11; // rax
  __int64 v12; // rcx
  size_t v13; // rsi
  __int64 v14; // rdi
  size_t v15; // rdx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  __int8 *src; // [rsp+8h] [rbp-88h]
  size_t v19; // [rsp+18h] [rbp-78h] BYREF
  __int8 *v20; // [rsp+20h] [rbp-70h] BYREF
  size_t n; // [rsp+28h] [rbp-68h]
  _BYTE v22[16]; // [rsp+30h] [rbp-60h] BYREF
  const char *v23; // [rsp+40h] [rbp-50h] BYREF
  size_t v24; // [rsp+48h] [rbp-48h]
  _QWORD v25[8]; // [rsp+50h] [rbp-40h] BYREF

  v1 = a1 + 8;
  v20 = v22;
  n = 0;
  v22[0] = 0;
  v2 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v2;
  if ( v2 != 61 )
  {
    if ( v2 != 65 )
    {
      v3 = *(_QWORD *)(a1 + 56);
      LOWORD(v25[0]) = 259;
      v23 = "unknown target property";
      v4 = sub_38814C0(v1, v3, (__int64)&v23);
      goto LABEL_4;
    }
    *(_DWORD *)(a1 + 64) = sub_3887100(v1);
    if ( !(unsigned __int8)sub_388AF10(a1, 3, "expected '=' after target datalayout") )
    {
      v4 = sub_388B0A0(a1, (unsigned __int64 *)&v20);
      if ( !(_BYTE)v4 )
      {
        if ( !*(_QWORD *)(a1 + 1456) )
          sub_1632B30(*(_QWORD *)(a1 + 176), v20, n);
        goto LABEL_4;
      }
    }
LABEL_8:
    v4 = 1;
    goto LABEL_4;
  }
  *(_DWORD *)(a1 + 64) = sub_3887100(v1);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after target triple") )
    goto LABEL_8;
  v4 = sub_388B0A0(a1, (unsigned __int64 *)&v20);
  if ( (_BYTE)v4 )
    goto LABEL_8;
  v6 = v20;
  v7 = *(_QWORD **)(a1 + 176);
  if ( !v20 )
  {
    LOBYTE(v25[0]) = 0;
    v15 = 0;
    v23 = (const char *)v25;
    v10 = (char *)v7[30];
LABEL_25:
    v7[31] = v15;
    v10[v15] = 0;
    v11 = v23;
    goto LABEL_22;
  }
  v8 = n;
  v23 = (const char *)v25;
  v19 = n;
  if ( n > 0xF )
  {
    src = v20;
    v16 = sub_22409D0((__int64)&v23, &v19, 0);
    v6 = src;
    v23 = (const char *)v16;
    v17 = (_QWORD *)v16;
    v25[0] = v19;
  }
  else
  {
    if ( n == 1 )
    {
      LOBYTE(v25[0]) = *v20;
      v9 = (const char *)v25;
      goto LABEL_18;
    }
    if ( !n )
    {
      v9 = (const char *)v25;
      goto LABEL_18;
    }
    v17 = v25;
  }
  memcpy(v17, v6, v8);
  v8 = v19;
  v9 = v23;
LABEL_18:
  v24 = v8;
  v9[v8] = 0;
  v10 = (char *)v7[30];
  v11 = v10;
  if ( v23 == (const char *)v25 )
  {
    v15 = v24;
    if ( v24 )
    {
      if ( v24 == 1 )
        *v10 = v25[0];
      else
        memcpy(v10, v25, v24);
      v15 = v24;
      v10 = (char *)v7[30];
    }
    goto LABEL_25;
  }
  v12 = v25[0];
  v13 = v24;
  if ( v10 == (char *)(v7 + 32) )
  {
    v7[30] = v23;
    v7[31] = v13;
    v7[32] = v12;
  }
  else
  {
    v14 = v7[32];
    v7[30] = v23;
    v7[31] = v13;
    v7[32] = v12;
    if ( v11 )
    {
      v23 = v11;
      v25[0] = v14;
      goto LABEL_22;
    }
  }
  v23 = (const char *)v25;
  v11 = (const char *)v25;
LABEL_22:
  v24 = 0;
  *v11 = 0;
  if ( v23 != (const char *)v25 )
    j_j___libc_free_0((unsigned __int64)v23);
LABEL_4:
  if ( v20 != v22 )
    j_j___libc_free_0((unsigned __int64)v20);
  return v4;
}

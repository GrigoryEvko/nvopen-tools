// Function: sub_120B4E0
// Address: 0x120b4e0
//
__int64 __fastcall sub_120B4E0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // r13
  int v5; // eax
  unsigned __int64 v6; // rsi
  unsigned int v7; // r13d
  _QWORD *v9; // rbx
  char *v10; // rdi
  __int64 v11; // rdx
  size_t v12; // rcx
  __int64 v13; // rsi
  size_t v14; // rdx
  _QWORD v15[2]; // [rsp+10h] [rbp-C0h] BYREF
  _QWORD v16[2]; // [rsp+20h] [rbp-B0h] BYREF
  _QWORD *v17; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v18; // [rsp+50h] [rbp-80h]
  const char *v19; // [rsp+60h] [rbp-70h] BYREF
  size_t n; // [rsp+68h] [rbp-68h]
  _QWORD src[2]; // [rsp+70h] [rbp-60h] BYREF
  __int64 v22; // [rsp+80h] [rbp-50h]
  __int64 v23; // [rsp+88h] [rbp-48h]
  __int64 v24; // [rsp+90h] [rbp-40h]

  v3 = a1 + 176;
  v15[0] = v16;
  v15[1] = 0;
  LOBYTE(v16[0]) = 0;
  v5 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v5;
  if ( v5 != 64 )
  {
    if ( v5 != 67 )
    {
      v6 = *(_QWORD *)(a1 + 232);
      LOWORD(v22) = 259;
      v7 = 1;
      v19 = "unknown target property";
      sub_11FD800(a1 + 176, v6, (__int64)&v19, 1);
      goto LABEL_4;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v3);
    if ( !(unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after target datalayout") )
    {
      *a3 = *(_QWORD *)(a1 + 232);
      v7 = sub_120B3D0(a1, a2);
      goto LABEL_4;
    }
LABEL_8:
    v7 = 1;
    goto LABEL_4;
  }
  *(_DWORD *)(a1 + 240) = sub_1205200(v3);
  if ( (unsigned __int8)sub_120AFE0(a1, 3, "expected '=' after target triple") )
    goto LABEL_8;
  v7 = sub_120B3D0(a1, (__int64)v15);
  if ( (_BYTE)v7 )
    goto LABEL_8;
  v9 = *(_QWORD **)(a1 + 344);
  v18 = 260;
  v17 = v15;
  sub_CC9F70((__int64)&v19, (void **)&v17);
  v10 = (char *)v9[29];
  if ( v19 == (const char *)src )
  {
    v14 = n;
    if ( n )
    {
      if ( n == 1 )
        *v10 = src[0];
      else
        memcpy(v10, src, n);
      v14 = n;
      v10 = (char *)v9[29];
    }
    v9[30] = v14;
    v10[v14] = 0;
    v10 = (char *)v19;
    goto LABEL_16;
  }
  v11 = src[0];
  v12 = n;
  if ( v10 == (char *)(v9 + 31) )
  {
    v9[29] = v19;
    v9[30] = v12;
    v9[31] = v11;
  }
  else
  {
    v13 = v9[31];
    v9[29] = v19;
    v9[30] = v12;
    v9[31] = v11;
    if ( v10 )
    {
      v19 = v10;
      src[0] = v13;
      goto LABEL_16;
    }
  }
  v19 = (const char *)src;
  v10 = (char *)src;
LABEL_16:
  n = 0;
  *v10 = 0;
  v9[33] = v22;
  v9[34] = v23;
  v9[35] = v24;
  if ( v19 != (const char *)src )
    j_j___libc_free_0(v19, src[0] + 1LL);
LABEL_4:
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0], v16[0] + 1LL);
  return v7;
}

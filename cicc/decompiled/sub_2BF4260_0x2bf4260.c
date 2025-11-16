// Function: sub_2BF4260
// Address: 0x2bf4260
//
__int64 __fastcall sub_2BF4260(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v4; // r15
  unsigned int v5; // r13d
  unsigned __int64 v6; // r12
  __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v10; // r8
  _QWORD *v11; // rdi
  size_t v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  char *v16; // rcx
  unsigned __int64 v17; // rsi
  __int64 v18; // r9
  int v19; // eax
  _QWORD *v20; // rdx
  _QWORD *v21; // rdi
  size_t v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rdi
  char *v25; // r15
  size_t v26; // rdx
  __int64 v27; // rdi
  unsigned int v28; // [rsp+4h] [rbp-BCh]
  _QWORD *v29; // [rsp+28h] [rbp-98h] BYREF
  int v30; // [rsp+30h] [rbp-90h] BYREF
  char v31; // [rsp+34h] [rbp-8Ch]
  unsigned int v32; // [rsp+38h] [rbp-88h]
  char v33; // [rsp+3Ch] [rbp-84h]
  _QWORD *v34; // [rsp+40h] [rbp-80h] BYREF
  size_t n; // [rsp+48h] [rbp-78h]
  _QWORD src[2]; // [rsp+50h] [rbp-70h] BYREF
  const char *v37; // [rsp+60h] [rbp-60h] BYREF
  char v38; // [rsp+80h] [rbp-40h]
  char v39; // [rsp+81h] [rbp-3Fh]

  result = (unsigned int)a2;
  v4 = HIDWORD(a2);
  v5 = 2 * a3;
  v6 = HIDWORD(a3);
  while ( (!(_BYTE)v4 || (_BYTE)v6) && v5 > (unsigned int)result )
  {
    v30 = result;
    v31 = v4;
    v32 = v5;
    v33 = v6;
    sub_2AD9B30((__int64 *)&v29, (__int64 *)a1, &v30);
    sub_2C3ACB0(v29);
    v8 = sub_2BF3F10(v29);
    v39 = 1;
    v9 = *(_QWORD **)(v8 + 120);
    v38 = 3;
    v37 = "vector.latch";
    sub_CA0F50((__int64 *)&v34, (void **)&v37);
    v11 = (_QWORD *)v9[2];
    if ( v34 == src )
    {
      v22 = n;
      if ( !n )
        goto LABEL_20;
      if ( n != 1 )
      {
        memcpy(v11, src, n);
        v22 = n;
        v11 = (_QWORD *)v9[2];
LABEL_20:
        v9[3] = v22;
        *((_BYTE *)v11 + v22) = 0;
        v11 = v34;
        goto LABEL_10;
      }
      *(_BYTE *)v11 = src[0];
      v26 = n;
      v27 = v9[2];
      v9[3] = n;
      *(_BYTE *)(v27 + v26) = 0;
      v11 = v34;
    }
    else
    {
      v12 = n;
      if ( v11 == v9 + 4 )
      {
        v9[2] = v34;
        v23 = src[0];
        v9[3] = v12;
        v9[4] = v23;
      }
      else
      {
        v9[2] = v34;
        v13 = src[0];
        v14 = v9[4];
        v9[3] = v12;
        v9[4] = v13;
        if ( v11 )
        {
          v34 = v11;
          src[0] = v14;
          goto LABEL_10;
        }
      }
      v34 = src;
      v11 = src;
    }
LABEL_10:
    n = 0;
    *(_BYTE *)v11 = 0;
    if ( v34 != src )
      j_j___libc_free_0((unsigned __int64)v34);
    v15 = *(unsigned int *)(a1 + 96);
    v16 = (char *)&v29;
    v17 = *(_QWORD *)(a1 + 88);
    v18 = v15 + 1;
    v19 = *(_DWORD *)(a1 + 96);
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 100) )
    {
      v24 = a1 + 88;
      if ( v17 > (unsigned __int64)&v29 || (unsigned __int64)&v29 >= v17 + 8 * v15 )
      {
        sub_2AC3D10(v24, v15 + 1, v15, (__int64)&v29, v10, v18);
        v15 = *(unsigned int *)(a1 + 96);
        v17 = *(_QWORD *)(a1 + 88);
        v16 = (char *)&v29;
        v19 = *(_DWORD *)(a1 + 96);
      }
      else
      {
        v25 = (char *)&v29 - v17;
        sub_2AC3D10(v24, v15 + 1, v15, (__int64)&v29 - v17, v10, v18);
        v17 = *(_QWORD *)(a1 + 88);
        v15 = *(unsigned int *)(a1 + 96);
        v16 = &v25[v17];
        v19 = *(_DWORD *)(a1 + 96);
      }
    }
    v20 = (_QWORD *)(v17 + 8 * v15);
    if ( v20 )
    {
      *v20 = *(_QWORD *)v16;
      *(_QWORD *)v16 = 0;
      v19 = *(_DWORD *)(a1 + 96);
    }
    v21 = v29;
    LOBYTE(v4) = v33;
    *(_DWORD *)(a1 + 96) = v19 + 1;
    result = v32;
    if ( v21 )
    {
      v28 = v32;
      sub_2BF1F00((__int64)v21, v17, (__int64)v20, (__int64)v16, v10, v18);
      j_j___libc_free_0((unsigned __int64)v21);
      result = v28;
    }
  }
  return result;
}

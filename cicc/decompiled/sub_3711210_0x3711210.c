// Function: sub_3711210
// Address: 0x3711210
//
unsigned __int64 *__fastcall sub_3711210(unsigned __int64 *a1, _QWORD *a2, __int64 a3, unsigned __int64 *a4)
{
  _QWORD *v4; // r15
  bool v6; // zf
  __int64 v7; // r13
  __int64 v8; // rax
  int v9; // r9d
  unsigned int v10; // ecx
  unsigned __int32 v11; // eax
  unsigned __int32 v12; // r13d
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  __int64 v16; // r13
  unsigned int *v17; // r13
  unsigned int *v18; // rbx
  unsigned __int64 v19; // rax
  char *v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // r12
  int v23; // r13d
  unsigned int (*v24)(void); // rax
  unsigned int *v25; // r13
  unsigned int *i; // rbx
  unsigned __int32 v27; // r12d
  unsigned __int32 v28; // [rsp+14h] [rbp-CCh]
  unsigned __int64 v29; // [rsp+28h] [rbp-B8h]
  int v30; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v31; // [rsp+28h] [rbp-B8h]
  unsigned int v32; // [rsp+3Ch] [rbp-A4h] BYREF
  unsigned __int64 v33; // [rsp+40h] [rbp-A0h] BYREF
  unsigned __int64 v34; // [rsp+48h] [rbp-98h] BYREF
  __m128i v35[2]; // [rsp+50h] [rbp-90h] BYREF
  char v36; // [rsp+70h] [rbp-70h]
  char v37; // [rsp+71h] [rbp-6Fh]
  __int64 v38[4]; // [rsp+80h] [rbp-60h] BYREF
  char v39; // [rsp+A0h] [rbp-40h]
  char v40; // [rsp+A1h] [rbp-3Fh]

  v4 = a2 + 2;
  v6 = a2[9] == 0;
  v37 = 1;
  v7 = a2[7];
  v35[0].m128i_i64[0] = (__int64)"NumStrings";
  v36 = 3;
  if ( v6 )
  {
    v21 = a2[8];
    if ( v21 && !v7 )
    {
      v22 = (__int64)(a4[2] - a4[1]) >> 2;
      v23 = v22;
      v24 = *(unsigned int (**)(void))(**(_QWORD **)(v21 + 24) + 16LL);
      if ( (char *)v24 != (char *)sub_3700C70 )
      {
        v27 = _byteswap_ulong(v22);
        if ( v24() != 1 )
          v23 = v27;
      }
      LODWORD(v34) = v23;
      sub_3719260(v38, v21, &v34, 4);
      if ( (v38[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
      {
        v33 = 0;
        v38[0] = v38[0] & 0xFFFFFFFFFFFFFFFELL | 1;
        sub_9C6670((__int64 *)&v33, v38);
        sub_9C66B0(v38);
        v14 = v33 & 0xFFFFFFFFFFFFFFFELL;
LABEL_38:
        if ( v14 )
          goto LABEL_11;
LABEL_22:
        *a1 = 1;
        return a1;
      }
      v38[0] = 0;
      sub_9C66B0(v38);
      v25 = (unsigned int *)a4[2];
      for ( i = (unsigned int *)a4[1]; v25 != i; ++i )
      {
        v40 = 1;
        v38[0] = (__int64)"Strings";
        v39 = 3;
        sub_37011E0(&v34, v4, i, v38);
        v19 = v34 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          goto LABEL_37;
        v34 = 0;
        sub_9C66B0((__int64 *)&v34);
      }
LABEL_21:
      v38[0] = 0;
      v33 = 1;
      sub_9C66B0(v38);
      goto LABEL_22;
    }
  }
  else if ( !v7 && !a2[8] )
  {
    v16 = a4[2] - a4[1];
    sub_370BB40(v4, v35);
    (*(void (__fastcall **)(_QWORD, _QWORD, __int64))(*(_QWORD *)a2[9] + 8LL))(a2[9], (unsigned int)(v16 >> 2), 4);
    if ( a2[9] && !a2[7] && !a2[8] )
      a2[10] += 4LL;
    v17 = (unsigned int *)a4[1];
    v18 = (unsigned int *)a4[2];
    if ( v17 != v18 )
    {
      while ( 1 )
      {
        v40 = 1;
        v38[0] = (__int64)"Strings";
        v39 = 3;
        sub_37011E0(&v34, v4, v17, v38);
        v19 = v34 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
        if ( v18 == ++v17 )
          goto LABEL_21;
      }
LABEL_37:
      v33 = 0;
      v34 = v19 | 1;
      sub_9C6670((__int64 *)&v33, &v34);
      sub_9C66B0((__int64 *)&v34);
      v14 = v33 & 0xFFFFFFFFFFFFFFFELL;
      goto LABEL_38;
    }
    goto LABEL_21;
  }
  v38[0] = 0;
  v38[1] = 0;
  sub_1254950(&v34, v7, (__int64)v38, 4u);
  v8 = v34;
  v34 = 0;
  v29 = v8 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v8 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    sub_9C66B0((__int64 *)&v34);
    v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(v7 + 24) + 16LL))(*(_QWORD *)(v7 + 24));
    v10 = *(_DWORD *)v38[0];
    v34 = 0;
    v11 = _byteswap_ulong(v10);
    if ( v9 == 1 )
      v11 = v10;
    v12 = v11;
    v28 = v11;
    sub_9C66B0((__int64 *)&v34);
    v38[0] = 0;
    sub_9C66B0(v38);
    if ( v12 )
    {
      v30 = 0;
      while ( 1 )
      {
        v32 = 0;
        v38[0] = (__int64)"Strings";
        v40 = 1;
        v39 = 3;
        sub_37011E0(&v34, v4, &v32, v38);
        v13 = v34 & 0xFFFFFFFFFFFFFFFELL;
        if ( (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
          break;
        v20 = (char *)a4[2];
        if ( v20 == (char *)a4[3] )
        {
          sub_370C1B0(a4 + 1, v20, &v32);
        }
        else
        {
          if ( v20 )
          {
            *(_DWORD *)v20 = v32;
            v20 = (char *)a4[2];
          }
          a4[2] = (unsigned __int64)(v20 + 4);
        }
        if ( ++v30 == v28 )
          goto LABEL_21;
      }
      v31 = v34 & 0xFFFFFFFFFFFFFFFELL;
      v34 = 0;
      v33 = v13 | 1;
      sub_9C66B0((__int64 *)&v34);
      v14 = v31;
      goto LABEL_11;
    }
    goto LABEL_21;
  }
  sub_9C66B0((__int64 *)&v34);
  v38[0] = 0;
  v33 = v29 | 1;
  sub_9C66B0(v38);
  v14 = v29;
LABEL_11:
  *a1 = v14 | 1;
  return a1;
}

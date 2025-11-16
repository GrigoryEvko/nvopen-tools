// Function: sub_1298D60
// Address: 0x1298d60
//
__int64 __fastcall sub_1298D60(__int64 a1, __int64 a2, unsigned __int16 a3, const char *a4)
{
  const char *v4; // r13
  __int64 v7; // r8
  __int64 *v8; // r14
  __int64 i; // rax
  char v10; // al
  __int64 v11; // rdi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  _BYTE *v19; // rdi
  __int64 v21; // rdi
  _BYTE *v22; // [rsp+0h] [rbp-B0h]
  __int64 v23; // [rsp+0h] [rbp-B0h]
  __int64 v24; // [rsp+8h] [rbp-A8h]
  __int64 v25; // [rsp+8h] [rbp-A8h]
  char *v26; // [rsp+18h] [rbp-98h] BYREF
  _BYTE v27[16]; // [rsp+20h] [rbp-90h] BYREF
  __int16 v28; // [rsp+30h] [rbp-80h]
  _QWORD *v29; // [rsp+40h] [rbp-70h] BYREF
  __int64 v30; // [rsp+48h] [rbp-68h]
  _QWORD v31[2]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE *v32; // [rsp+60h] [rbp-50h] BYREF
  __int64 v33; // [rsp+68h] [rbp-48h]
  _BYTE v34[64]; // [rsp+70h] [rbp-40h] BYREF

  v4 = a4;
  v29 = v31;
  v30 = 0;
  LOBYTE(v31[0]) = 0;
  if ( a3 == 10227 )
  {
    v7 = 7;
    a4 = "__isnan";
    goto LABEL_6;
  }
  if ( a3 <= 0x27F3u )
  {
    if ( a3 == 10214 )
    {
      v7 = 8;
      a4 = "__finite";
    }
    else
    {
      if ( a3 != 10219 )
        goto LABEL_7;
      v7 = 7;
      a4 = "__isinf";
    }
    goto LABEL_6;
  }
  v7 = 9;
  a4 = "__signbit";
  if ( a3 == 15752 )
LABEL_6:
    sub_2241130(&v29, 0, 0, a4, v7);
LABEL_7:
  v8 = (__int64 *)*((_QWORD *)v4 + 2);
  for ( i = *v8; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v10 = *(_BYTE *)(i + 160);
  if ( v10 != 2 )
  {
    v11 = *(_QWORD *)(a2 + 40);
    v32 = v34;
    v33 = 0x200000000LL;
    if ( v10 == 8 )
    {
      v12 = sub_16432F0(v11);
      v13 = (unsigned int)v33;
      if ( (unsigned int)v33 < HIDWORD(v33) )
        goto LABEL_12;
    }
    else
    {
      v12 = sub_16432B0(v11);
      v13 = (unsigned int)v33;
      if ( (unsigned int)v33 < HIDWORD(v33) )
        goto LABEL_12;
    }
    goto LABEL_20;
  }
  if ( v30 == 0x3FFFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"basic_string::append");
  sub_2241490(&v29, "f", 1, a4);
  v21 = *(_QWORD *)(a2 + 40);
  v32 = v34;
  v33 = 0x200000000LL;
  v12 = sub_16432A0(v21);
  v13 = (unsigned int)v33;
  if ( (unsigned int)v33 >= HIDWORD(v33) )
  {
LABEL_20:
    v23 = v12;
    sub_16CD150(&v32, v34, 0, 8);
    v13 = (unsigned int)v33;
    v12 = v23;
  }
LABEL_12:
  *(_QWORD *)&v32[8 * v13] = v12;
  v14 = *(_QWORD *)(a2 + 40);
  LODWORD(v33) = v33 + 1;
  v22 = v32;
  v24 = (unsigned int)v33;
  v15 = sub_1643350(v14);
  v25 = sub_1644EA0(v15, v22, v24, 0);
  v28 = 257;
  v26 = sub_128F980(a2, (__int64)v8);
  v16 = sub_1632190(**(_QWORD **)(a2 + 32), v29, v30, v25);
  v17 = *(_QWORD *)(*(_QWORD *)v16 + 24LL);
  v18 = sub_1285290((__int64 *)(a2 + 48), v17, v16, (int)&v26, 1, (__int64)v27, 0);
  v19 = v32;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v18;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v19 != v34 )
    _libc_free(v19, v17);
  if ( v29 != v31 )
    j_j___libc_free_0(v29, v31[0] + 1LL);
  return a1;
}

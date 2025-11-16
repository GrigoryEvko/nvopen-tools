// Function: sub_939370
// Address: 0x939370
//
__int64 __fastcall sub_939370(__int64 a1, __int64 a2, __int64 a3, const char *a4)
{
  const char *v4; // r13
  __int64 v7; // r8
  __int64 *v8; // r14
  __int64 i; // rax
  char v10; // al
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned __int64 v16; // rsi
  int v17; // edx
  __int64 v18; // rax
  _BYTE *v19; // rdi
  __int64 v21; // rdi
  _BYTE *v22; // [rsp+0h] [rbp-C0h]
  __int64 v23; // [rsp+0h] [rbp-C0h]
  __int64 v24; // [rsp+8h] [rbp-B8h]
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __m128i *v26; // [rsp+18h] [rbp-A8h] BYREF
  _QWORD *v27; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-98h]
  _QWORD v29[2]; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v30; // [rsp+40h] [rbp-80h] BYREF
  __int64 v31; // [rsp+48h] [rbp-78h]
  _BYTE v32[16]; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v33[32]; // [rsp+60h] [rbp-60h] BYREF
  __int16 v34; // [rsp+80h] [rbp-40h]

  v4 = a4;
  v27 = v29;
  v28 = 0;
  LOBYTE(v29[0]) = 0;
  if ( (_WORD)a3 == 10227 )
  {
    v7 = 7;
    a4 = "__isnan";
    goto LABEL_6;
  }
  if ( (unsigned __int16)a3 <= 0x27F3u )
  {
    if ( (_WORD)a3 == 10214 )
    {
      v7 = 8;
      a4 = "__finite";
    }
    else
    {
      if ( (_WORD)a3 != 10219 )
        goto LABEL_7;
      v7 = 7;
      a4 = "__isinf";
    }
    goto LABEL_6;
  }
  v7 = 9;
  a4 = "__signbit";
  if ( (_WORD)a3 == 15752 )
LABEL_6:
    sub_2241130(&v27, 0, 0, a4, v7);
LABEL_7:
  v8 = (__int64 *)*((_QWORD *)v4 + 2);
  for ( i = *v8; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v10 = *(_BYTE *)(i + 160);
  if ( v10 == 2 )
  {
    if ( v28 == 0x3FFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"basic_string::append");
    sub_2241490(&v27, "f", 1, a4);
    v21 = *(_QWORD *)(a2 + 40);
    v30 = v32;
    v31 = 0x200000000LL;
    v12 = sub_BCB160(v21);
  }
  else
  {
    v11 = *(_QWORD *)(a2 + 40);
    v30 = v32;
    v31 = 0x200000000LL;
    if ( v10 == 8 )
      v12 = sub_BCB1B0(v11, &v30, a3, 0x200000000LL);
    else
      v12 = sub_BCB170(v11);
  }
  v13 = (unsigned int)v31;
  if ( (unsigned __int64)(unsigned int)v31 + 1 > HIDWORD(v31) )
  {
    v23 = v12;
    sub_C8D5F0(&v30, v32, (unsigned int)v31 + 1LL, 8);
    v13 = (unsigned int)v31;
    v12 = v23;
  }
  *(_QWORD *)&v30[8 * v13] = v12;
  v14 = *(_QWORD *)(a2 + 40);
  LODWORD(v31) = v31 + 1;
  v22 = v30;
  v24 = (unsigned int)v31;
  v15 = sub_BCB2D0(v14);
  v25 = sub_BCF480(v15, v22, v24, 0);
  v34 = 257;
  v26 = sub_92F410(a2, (__int64)v8);
  v16 = sub_BA8CA0(**(_QWORD **)(a2 + 32), v27, v28, v25);
  v18 = sub_921880((unsigned int **)(a2 + 48), v16, v17, (int)&v26, 1, (__int64)v33, 0);
  v19 = v30;
  *(_DWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = v18;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v19 != v32 )
    _libc_free(v19, v16);
  if ( v27 != v29 )
    j_j___libc_free_0(v27, v29[0] + 1LL);
  return a1;
}

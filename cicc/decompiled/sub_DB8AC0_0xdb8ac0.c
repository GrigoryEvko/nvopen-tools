// Function: sub_DB8AC0
// Address: 0xdb8ac0
//
__int64 __fastcall sub_DB8AC0(
        __int64 a1,
        int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        char a7,
        unsigned __int8 a8)
{
  unsigned int v12; // ebx
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rsi
  char *v18; // rdi
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // [rsp+8h] [rbp-C8h]
  __int64 v24; // [rsp+8h] [rbp-C8h]
  __int64 v25; // [rsp+40h] [rbp-90h] BYREF
  __int64 v26; // [rsp+48h] [rbp-88h]
  __int64 v27; // [rsp+50h] [rbp-80h]
  char v28; // [rsp+58h] [rbp-78h]
  char *v29; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v30; // [rsp+68h] [rbp-68h]
  _BYTE v31[32]; // [rsp+70h] [rbp-60h] BYREF
  char v32; // [rsp+90h] [rbp-40h]

  v12 = a6;
  sub_D97FD0((__int64)&v25, a3, a4, a5, a6, a7);
  v15 = a1 + 48;
  if ( v32 )
  {
    v16 = v25;
    v17 = v30;
    *(_QWORD *)(a1 + 32) = v15;
    *(_QWORD *)a1 = v16;
    *(_QWORD *)(a1 + 8) = v26;
    *(_QWORD *)(a1 + 16) = v27;
    *(_BYTE *)(a1 + 24) = v28;
    *(_QWORD *)(a1 + 40) = 0x400000000LL;
    if ( (_DWORD)v17 )
    {
      v17 = (__int64)&v29;
      sub_D915C0(a1 + 32, (__int64)&v29, v15, v13, v23, v14);
      if ( !v32 )
        return a1;
    }
  }
  else
  {
    sub_DF25E0((unsigned int)&v25, a2, a3, a4, a5, v12, a7, a8);
    v17 = a4;
    sub_DB8730(a3, a4, a5, v12, a7, a8, &v25);
    v22 = v30;
    *(_QWORD *)a1 = v25;
    *(_QWORD *)(a1 + 8) = v26;
    *(_QWORD *)(a1 + 16) = v27;
    *(_BYTE *)(a1 + 24) = v28;
    *(_QWORD *)(a1 + 32) = a1 + 48;
    *(_QWORD *)(a1 + 40) = 0x400000000LL;
    if ( (_DWORD)v22 )
    {
      v17 = (__int64)&v29;
      sub_D91460(a1 + 32, &v29, v24, v22, v20, v21);
      v18 = v29;
      if ( v29 == v31 )
        return a1;
      goto LABEL_4;
    }
  }
  v18 = v29;
  if ( v29 != v31 )
LABEL_4:
    _libc_free(v18, v17);
  return a1;
}

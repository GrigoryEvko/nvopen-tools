// Function: sub_24C54D0
// Address: 0x24c54d0
//
__int64 __fastcall sub_24C54D0(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, const char *a5)
{
  _QWORD *v8; // rax
  __int64 v9; // r12
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // rsi
  const char *v14; // rax
  size_t v15; // r8
  __int64 v16; // rdx
  unsigned __int8 v17; // si
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  _QWORD *v26; // rdi
  __int64 v27; // rax
  size_t na; // [rsp+8h] [rbp-98h]
  _QWORD *v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  __int64 v32[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v33[2]; // [rsp+30h] [rbp-70h] BYREF
  const char *v34; // [rsp+40h] [rbp-60h] BYREF
  __int64 v35; // [rsp+48h] [rbp-58h]
  __int64 v36; // [rsp+50h] [rbp-50h] BYREF
  char v37; // [rsp+60h] [rbp-40h]
  char v38; // [rsp+61h] [rbp-3Fh]

  v30 = sub_BCD420(a4, a2);
  v38 = 1;
  v31 = sub_AD6530((__int64)v30, a2);
  v34 = "__sancov_gen_";
  v37 = 3;
  BYTE4(v32[0]) = 0;
  v8 = sub_BD2C40(88, unk_3F0FAE8);
  v9 = (__int64)v8;
  if ( v8 )
    sub_B30000((__int64)v8, *(_QWORD *)(a1 + 512), v30, 0, 8, v31, (__int64)&v34, 0, 0, v32[0], 0);
  v10 = *(unsigned int *)(a1 + 604);
  if ( (unsigned int)v10 > 8 )
    goto LABEL_35;
  v24 = 292;
  if ( _bittest64(&v24, v10) )
    goto LABEL_7;
  if ( (_DWORD)v10 != 3 )
  {
LABEL_35:
    if ( (unsigned __int8)sub_B2F6B0(a3) )
      goto LABEL_7;
  }
  v13 = sub_29F3CB0(a3, a1 + 552);
  if ( v13 )
    sub_B2F990(v9, v13, v11, v12);
LABEL_7:
  v32[0] = (__int64)v33;
  v14 = (const char *)strlen(a5);
  v34 = v14;
  v15 = (size_t)v14;
  if ( (unsigned __int64)v14 > 0xF )
  {
    na = (size_t)v14;
    v25 = sub_22409D0((__int64)v32, (unsigned __int64 *)&v34, 0);
    v15 = na;
    v32[0] = v25;
    v26 = (_QWORD *)v25;
    v33[0] = v34;
  }
  else
  {
    if ( v14 == (const char *)1 )
    {
      LOBYTE(v33[0]) = *a5;
      v16 = (__int64)v33;
      goto LABEL_10;
    }
    if ( !v14 )
    {
      v16 = (__int64)v33;
      goto LABEL_10;
    }
    v26 = v33;
  }
  memcpy(v26, a5, v15);
  v14 = v34;
  v16 = v32[0];
LABEL_10:
  v32[1] = (__int64)v14;
  v14[v16] = 0;
  sub_24C5360((__int64)&v34, a1, v32);
  sub_B31A00(v9, (__int64)v34, v35);
  if ( v34 != (const char *)&v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  if ( (_QWORD *)v32[0] != v33 )
    j_j___libc_free_0(v32[0]);
  v17 = -1;
  v34 = (const char *)sub_9208B0(*(_QWORD *)(a1 + 616), (__int64)a4);
  v18 = (unsigned __int64)(v34 + 7) >> 3;
  v35 = v19;
  if ( v18 )
  {
    _BitScanReverse64(&v18, v18);
    v17 = 63 - (v18 ^ 0x3F);
  }
  sub_B2F770(v9, v17);
  if ( *(_QWORD *)(v9 + 48) )
  {
    v22 = *(unsigned int *)(a1 + 848);
    if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 852) )
    {
      sub_C8D5F0(a1 + 840, (const void *)(a1 + 856), v22 + 1, 8u, v20, v21);
      v22 = *(unsigned int *)(a1 + 848);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 840) + 8 * v22) = v9;
    ++*(_DWORD *)(a1 + 848);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 672);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 676) )
    {
      sub_C8D5F0(a1 + 664, (const void *)(a1 + 680), v27 + 1, 8u, v20, v21);
      v27 = *(unsigned int *)(a1 + 672);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 664) + 8 * v27) = v9;
    ++*(_DWORD *)(a1 + 672);
  }
  return v9;
}

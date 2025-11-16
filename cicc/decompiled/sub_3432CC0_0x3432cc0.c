// Function: sub_3432CC0
// Address: 0x3432cc0
//
__int64 __fastcall sub_3432CC0(__int64 a1, __int64 a2, __int64 *a3, __m128i a4)
{
  _WORD *v5; // rdx
  _QWORD *v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rbx
  _BYTE *v12; // rdx
  __int64 v13; // rax
  int v14; // edx
  unsigned int *v15; // rdx
  _BYTE *v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rbx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __m128i v23; // [rsp+20h] [rbp-100h] BYREF
  __int64 v24; // [rsp+30h] [rbp-F0h] BYREF
  _BYTE *v25; // [rsp+40h] [rbp-E0h] BYREF
  __int64 v26; // [rsp+48h] [rbp-D8h]
  _BYTE v27[32]; // [rsp+50h] [rbp-D0h] BYREF
  _QWORD v28[3]; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+88h] [rbp-98h]
  __int64 v30; // [rsp+90h] [rbp-90h]
  __int64 v31; // [rsp+98h] [rbp-88h]
  __int64 v32; // [rsp+A0h] [rbp-80h]
  _QWORD v33[14]; // [rsp+B0h] [rbp-70h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v31 = 0x100000000LL;
  v32 = a1;
  v28[0] = &unk_49DD210;
  v28[1] = 0;
  v28[2] = 0;
  v29 = 0;
  v30 = 0;
  sub_CB5980((__int64)v28, 0, 0, 0);
  v5 = (_WORD *)v30;
  if ( (unsigned __int64)(v29 - v30) <= 2 )
  {
    v6 = (_QWORD *)sub_CB6200((__int64)v28, "SU(", 3u);
  }
  else
  {
    *(_BYTE *)(v30 + 2) = 40;
    v6 = v28;
    *v5 = 21843;
    v30 += 3;
  }
  v7 = sub_CB59D0((__int64)v6, *((unsigned int *)a3 + 50));
  v10 = *(_QWORD *)(v7 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v7 + 24) - v10) <= 2 )
  {
    sub_CB6200(v7, (unsigned __int8 *)"): ", 3u);
    v11 = *a3;
    if ( v11 )
      goto LABEL_5;
LABEL_22:
    v21 = v30;
    if ( (unsigned __int64)(v29 - v30) <= 0xC )
    {
      sub_CB6200((__int64)v28, "CROSS RC COPY", 0xDu);
    }
    else
    {
      *(_DWORD *)(v30 + 8) = 1347371808;
      *(_QWORD *)v21 = 0x43522053534F5243LL;
      *(_BYTE *)(v21 + 12) = 89;
      v30 += 13;
    }
    goto LABEL_15;
  }
  *(_BYTE *)(v10 + 2) = 32;
  *(_WORD *)v10 = 14889;
  *(_QWORD *)(v7 + 32) += 3LL;
  v11 = *a3;
  if ( !v11 )
    goto LABEL_22;
LABEL_5:
  v12 = v27;
  v25 = v27;
  v26 = 0x400000000LL;
  v13 = 0;
  while ( 1 )
  {
    *(_QWORD *)&v12[8 * v13] = v11;
    v14 = *(_DWORD *)(v11 + 64);
    v13 = (unsigned int)(v26 + 1);
    LODWORD(v26) = v26 + 1;
    if ( !v14 )
      break;
    v15 = (unsigned int *)(*(_QWORD *)(v11 + 40) + 40LL * (unsigned int)(v14 - 1));
    v11 = *(_QWORD *)v15;
    if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v15 + 48LL) + 16LL * v15[2]) != 262 )
      break;
    if ( v13 + 1 > (unsigned __int64)HIDWORD(v26) )
    {
      sub_C8D5F0((__int64)&v25, v27, v13 + 1, 8u, v8, v9);
      v13 = (unsigned int)v26;
    }
    v12 = v25;
  }
  while ( 1 )
  {
    v16 = v25;
    if ( !(_DWORD)v13 )
      break;
    v17 = *(_QWORD *)&v25[8 * v13 - 8];
    v18 = *(_QWORD *)(a2 + 592);
    sub_3418C90(&v23, v17, v18);
    v33[6] = &v23;
    memset(&v33[1], 0, 32);
    v33[5] = 0x100000000LL;
    v33[0] = &unk_49DD210;
    sub_CB5980((__int64)v33, 0, 0, 0);
    sub_341C3A0(v17, (const char *)v33, v18, a4);
    v33[0] = &unk_49DD210;
    sub_CB5840((__int64)v33);
    sub_CB6200((__int64)v28, (unsigned __int8 *)v23.m128i_i64[0], v23.m128i_u64[1]);
    if ( (__int64 *)v23.m128i_i64[0] != &v24 )
      j_j___libc_free_0(v23.m128i_u64[0]);
    LODWORD(v26) = v26 - 1;
    if ( !(_DWORD)v26 )
    {
      v16 = v25;
      break;
    }
    v20 = v30;
    if ( (unsigned __int64)(v29 - v30) <= 4 )
    {
      sub_CB6200((__int64)v28, "\n    ", 5u);
      v13 = (unsigned int)v26;
    }
    else
    {
      *(_DWORD *)v30 = 538976266;
      *(_BYTE *)(v20 + 4) = 32;
      v13 = (unsigned int)v26;
      v30 += 5;
    }
  }
  if ( v16 != v27 )
    _libc_free((unsigned __int64)v16);
LABEL_15:
  v28[0] = &unk_49DD210;
  sub_CB5840((__int64)v28);
  return a1;
}

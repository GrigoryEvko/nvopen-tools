// Function: sub_EB41F0
// Address: 0xeb41f0
//
__int64 __fastcall sub_EB41F0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rax
  _BYTE *v15; // rsi
  __int64 *v16; // rbx
  _QWORD *v17; // rdi
  _QWORD *v18; // rdi
  unsigned __int64 v19; // rbx
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // [rsp+10h] [rbp-70h] BYREF
  __int64 v24; // [rsp+18h] [rbp-68h] BYREF
  __int64 v25[4]; // [rsp+20h] [rbp-60h] BYREF
  char v26; // [rsp+40h] [rbp-40h]
  char v27; // [rsp+41h] [rbp-3Fh]

  v8 = a3[4];
  if ( (unsigned __int64)(a3[3] - v8) <= 5 )
  {
    sub_CB6200((__int64)a3, ".endr\n", 6u);
  }
  else
  {
    *(_DWORD *)v8 = 1684956462;
    *(_WORD *)(v8 + 4) = 2674;
    a3[4] += 6LL;
  }
  v27 = 1;
  v25[0] = (__int64)"<instantiation>";
  v9 = a3[6];
  v26 = 3;
  sub_C7DE20(&v23, *(const void **)v9, *(_QWORD *)(v9 + 8), (__int64)v25, a5, a6);
  v10 = sub_ECD7B0(a1);
  v11 = (__int64)(*(_QWORD *)(a1 + 328) - *(_QWORD *)(a1 + 320)) >> 3;
  v12 = sub_22077B0(32);
  v13 = v12;
  if ( v12 )
  {
    *(_QWORD *)v12 = a2;
    *(_DWORD *)(v12 + 8) = *(_DWORD *)(a1 + 304);
    v14 = sub_ECD6A0(v10);
    *(_QWORD *)(v13 + 24) = v11;
    *(_QWORD *)(v13 + 16) = v14;
  }
  v24 = v13;
  v15 = *(_BYTE **)(a1 + 376);
  if ( v15 == *(_BYTE **)(a1 + 384) )
  {
    sub_EA27B0(a1 + 368, v15, &v24);
  }
  else
  {
    if ( v15 )
    {
      *(_QWORD *)v15 = v13;
      v15 = *(_BYTE **)(a1 + 376);
    }
    *(_QWORD *)(a1 + 376) = v15 + 8;
  }
  v16 = *(__int64 **)(a1 + 248);
  v25[1] = 0;
  v25[2] = 0;
  v25[0] = v23;
  v17 = (_QWORD *)v16[1];
  v23 = 0;
  if ( v17 == (_QWORD *)v16[2] )
  {
    sub_C12520(v16, (__int64)v17, (__int64)v25);
    v18 = (_QWORD *)v16[1];
  }
  else
  {
    if ( v17 )
    {
      sub_C8EDF0(v17, v25);
      v17 = (_QWORD *)v16[1];
    }
    v18 = v17 + 3;
    v16[1] = (__int64)v18;
  }
  v19 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v18 - *v16) >> 3);
  sub_C8EE20(v25);
  v20 = *(_QWORD **)(a1 + 248);
  *(_DWORD *)(a1 + 304) = v19;
  v21 = *(_QWORD *)(*v20 + 24LL * (unsigned int)(v19 - 1));
  sub_1095BD0(a1 + 40, *(_QWORD *)(v21 + 8), *(_QWORD *)(v21 + 16) - *(_QWORD *)(v21 + 8), 0, 1);
  result = sub_EABFE0(a1);
  if ( v23 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v23 + 8LL))(v23);
  return result;
}

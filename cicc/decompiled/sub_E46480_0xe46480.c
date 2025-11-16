// Function: sub_E46480
// Address: 0xe46480
//
__int64 __fastcall sub_E46480(__int64 *a1, __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64); // rdx
  __int64 v3; // rdx
  _BYTE *v4; // r15
  __int64 v5; // r14
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rdi
  void **v9; // rsi
  __int64 result; // rax
  _BYTE *v11; // rbx
  _BYTE *v12; // r12
  _BYTE *v13; // rdi
  _QWORD *v14; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v15; // [rsp+18h] [rbp-1B8h]
  _QWORD v16[2]; // [rsp+20h] [rbp-1B0h] BYREF
  void *v17; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v18; // [rsp+38h] [rbp-198h]
  __int64 v19[2]; // [rsp+40h] [rbp-190h] BYREF
  _QWORD v20[2]; // [rsp+50h] [rbp-180h] BYREF
  __int64 v21; // [rsp+60h] [rbp-170h]
  int v22; // [rsp+68h] [rbp-168h]
  _QWORD *v23; // [rsp+70h] [rbp-160h] BYREF
  _QWORD v24[2]; // [rsp+80h] [rbp-150h] BYREF
  _QWORD *v25; // [rsp+90h] [rbp-140h]
  __int64 v26; // [rsp+98h] [rbp-138h]
  _QWORD v27[2]; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v28; // [rsp+B0h] [rbp-120h]
  __int64 v29; // [rsp+B8h] [rbp-118h]
  __int64 v30; // [rsp+C0h] [rbp-110h]
  _BYTE *v31; // [rsp+C8h] [rbp-108h]
  __int64 v32; // [rsp+D0h] [rbp-100h]
  _BYTE v33[248]; // [rsp+D8h] [rbp-F8h] BYREF

  v2 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 24LL);
  if ( v2 == sub_9C3610 )
  {
    v14 = v16;
    v21 = (__int64)&v14;
    v20[1] = 0x100000000LL;
    v15 = 0;
    LOBYTE(v16[0]) = 0;
    v18 = 0;
    v19[0] = 0;
    v19[1] = 0;
    v20[0] = 0;
    v17 = &unk_49DD210;
    sub_CB5980((__int64)&v17, 0, 0, 0);
    (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a2 + 16LL))(a2, &v17);
    v17 = &unk_49DD210;
    sub_CB5840((__int64)&v17);
  }
  else
  {
    ((void (__fastcall *)(_QWORD **))v2)(&v14);
  }
  v3 = a1[1];
  v4 = v14;
  v5 = v15;
  v6 = *(_BYTE **)(v3 + 16);
  v7 = *(_QWORD *)(v3 + 24);
  v19[0] = (__int64)v20;
  v17 = 0;
  v18 = 0;
  sub_E45AE0(v19, v6, (__int64)&v6[v7]);
  v23 = v24;
  v21 = -1;
  v22 = 0;
  sub_E45AE0((__int64 *)&v23, v4, (__int64)&v4[v5]);
  v8 = *a1;
  v9 = &v17;
  v32 = 0x400000000LL;
  v25 = v27;
  v26 = 0;
  LOBYTE(v27[0]) = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = v33;
  sub_E45B90(v8, (__int64)&v17);
  result = (unsigned int)v32;
  v11 = v31;
  v12 = &v31[48 * (unsigned int)v32];
  if ( v31 != v12 )
  {
    do
    {
      v12 -= 48;
      v13 = (_BYTE *)*((_QWORD *)v12 + 2);
      result = (__int64)(v12 + 32);
      if ( v13 != v12 + 32 )
      {
        v9 = (void **)(*((_QWORD *)v12 + 4) + 1LL);
        result = j_j___libc_free_0(v13, v9);
      }
    }
    while ( v11 != v12 );
    v12 = v31;
  }
  if ( v12 != v33 )
    result = _libc_free(v12, v9);
  if ( v28 )
    result = j_j___libc_free_0(v28, v30 - v28);
  if ( v25 != v27 )
    result = j_j___libc_free_0(v25, v27[0] + 1LL);
  if ( v23 != v24 )
    result = j_j___libc_free_0(v23, v24[0] + 1LL);
  if ( (_QWORD *)v19[0] != v20 )
    result = j_j___libc_free_0(v19[0], v20[0] + 1LL);
  if ( v14 != v16 )
    return j_j___libc_free_0(v14, v16[0] + 1LL);
  return result;
}

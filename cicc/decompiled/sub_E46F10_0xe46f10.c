// Function: sub_E46F10
// Address: 0xe46f10
//
__int64 __fastcall sub_E46F10(__int64 a1, __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64); // rax
  _BYTE *v4; // r15
  __int64 v5; // rdx
  char *v6; // rsi
  __int64 v7; // r14
  __int64 v8; // rdi
  char *(*v9)(); // rax
  __int64 v10; // rdi
  void **v11; // rsi
  __int64 result; // rax
  _BYTE *v13; // rbx
  _BYTE *v14; // r12
  _BYTE *v15; // rdi
  _QWORD *v16; // [rsp+10h] [rbp-1C0h] BYREF
  __int64 v17; // [rsp+18h] [rbp-1B8h]
  _QWORD v18[2]; // [rsp+20h] [rbp-1B0h] BYREF
  void *v19; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v20; // [rsp+38h] [rbp-198h]
  __int64 v21[2]; // [rsp+40h] [rbp-190h] BYREF
  _QWORD v22[2]; // [rsp+50h] [rbp-180h] BYREF
  __int64 v23; // [rsp+60h] [rbp-170h]
  int v24; // [rsp+68h] [rbp-168h]
  _QWORD *v25; // [rsp+70h] [rbp-160h] BYREF
  _QWORD v26[2]; // [rsp+80h] [rbp-150h] BYREF
  _QWORD *v27; // [rsp+90h] [rbp-140h]
  __int64 v28; // [rsp+98h] [rbp-138h]
  _QWORD v29[2]; // [rsp+A0h] [rbp-130h] BYREF
  __int64 v30; // [rsp+B0h] [rbp-120h]
  __int64 v31; // [rsp+B8h] [rbp-118h]
  __int64 v32; // [rsp+C0h] [rbp-110h]
  _BYTE *v33; // [rsp+C8h] [rbp-108h]
  __int64 v34; // [rsp+D0h] [rbp-100h]
  _BYTE v35[248]; // [rsp+D8h] [rbp-F8h] BYREF

  v3 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 24LL);
  if ( v3 == sub_9C3610 )
  {
    v16 = v18;
    v22[1] = 0x100000000LL;
    v23 = (__int64)&v16;
    v17 = 0;
    LOBYTE(v18[0]) = 0;
    v20 = 0;
    v21[0] = 0;
    v21[1] = 0;
    v22[0] = 0;
    v19 = &unk_49DD210;
    sub_CB5980((__int64)&v19, 0, 0, 0);
    (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a2 + 16LL))(a2, &v19);
    v19 = &unk_49DD210;
    sub_CB5840((__int64)&v19);
  }
  else
  {
    ((void (__fastcall *)(_QWORD **))v3)(&v16);
  }
  v4 = v16;
  v5 = 14;
  v6 = "Unknown buffer";
  v7 = v17;
  v8 = **(_QWORD **)(a1 + 8);
  v9 = *(char *(**)())(*(_QWORD *)v8 + 16LL);
  if ( v9 != sub_C1E8B0 )
    v6 = (char *)((__int64 (__fastcall *)(__int64, char *, __int64))v9)(v8, "Unknown buffer", 14);
  v19 = 0;
  v21[0] = (__int64)v22;
  v20 = 0;
  sub_E45AE0(v21, v6, (__int64)&v6[v5]);
  v25 = v26;
  v23 = -1;
  v24 = 0;
  sub_E45AE0((__int64 *)&v25, v4, (__int64)&v4[v7]);
  v10 = *(_QWORD *)a1;
  v11 = &v19;
  v34 = 0x400000000LL;
  v27 = v29;
  v28 = 0;
  LOBYTE(v29[0]) = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = v35;
  sub_E45B90(v10, (__int64)&v19);
  result = (unsigned int)v34;
  v13 = v33;
  v14 = &v33[48 * (unsigned int)v34];
  if ( v33 != v14 )
  {
    do
    {
      v14 -= 48;
      v15 = (_BYTE *)*((_QWORD *)v14 + 2);
      result = (__int64)(v14 + 32);
      if ( v15 != v14 + 32 )
      {
        v11 = (void **)(*((_QWORD *)v14 + 4) + 1LL);
        result = j_j___libc_free_0(v15, v11);
      }
    }
    while ( v13 != v14 );
    v14 = v33;
  }
  if ( v14 != v35 )
    result = _libc_free(v14, v11);
  if ( v30 )
    result = j_j___libc_free_0(v30, v32 - v30);
  if ( v27 != v29 )
    result = j_j___libc_free_0(v27, v29[0] + 1LL);
  if ( v25 != v26 )
    result = j_j___libc_free_0(v25, v26[0] + 1LL);
  if ( (_QWORD *)v21[0] != v22 )
    result = j_j___libc_free_0(v21[0], v22[0] + 1LL);
  if ( v16 != v18 )
    return j_j___libc_free_0(v16, v18[0] + 1LL);
  return result;
}

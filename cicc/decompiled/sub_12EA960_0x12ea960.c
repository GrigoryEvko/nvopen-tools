// Function: sub_12EA960
// Address: 0x12ea960
//
_QWORD *__fastcall sub_12EA960(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  void (__fastcall *v4)(__int64, __int64, _QWORD); // r15
  __int64 v5; // rax
  void (__fastcall *v6)(__int64, __int64, _QWORD); // r15
  __int64 v7; // rax
  __int64 v8; // r8
  void (__fastcall *v9)(__int64, __int64, _QWORD); // rbx
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rdi
  __int64 v15; // rax
  _QWORD *result; // rax
  __int64 v18; // [rsp+8h] [rbp-1A8h]
  _QWORD *v19; // [rsp+30h] [rbp-180h] BYREF
  _QWORD v20[3]; // [rsp+40h] [rbp-170h] BYREF
  int v21; // [rsp+5Ch] [rbp-154h]
  _QWORD v22[2]; // [rsp+70h] [rbp-140h] BYREF
  __int64 v23; // [rsp+80h] [rbp-130h] BYREF
  _QWORD v24[2]; // [rsp+B0h] [rbp-100h] BYREF
  _WORD v25[52]; // [rsp+C0h] [rbp-F0h]
  __int64 v26; // [rsp+128h] [rbp-88h]
  unsigned int v27; // [rsp+138h] [rbp-78h]
  __int64 v28; // [rsp+148h] [rbp-68h]
  __int64 v29; // [rsp+158h] [rbp-58h]
  __int64 v30; // [rsp+160h] [rbp-50h]
  __int64 v31; // [rsp+170h] [rbp-40h]

  v25[0] = 260;
  v24[0] = a3 + 240;
  sub_16E1010(&v19);
  v4 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  if ( *a4 )
    sub_1700880(v24);
  else
    sub_14A3CD0(v24);
  v5 = sub_14A4230(v24);
  v4(a2, v5, 0);
  if ( *(_QWORD *)v25 )
    (*(void (__fastcall **)(_QWORD *, _QWORD *, __int64))v25)(v24, v24, 3);
  sub_16E1010(v22);
  sub_14A04B0(v24, v22);
  if ( (__int64 *)v22[0] != &v23 )
    j_j___libc_free_0(v22[0], v23 + 1);
  sub_149CBC0(v24);
  v6 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v7 = sub_22077B0(368);
  v8 = v7;
  if ( v7 )
  {
    v18 = v7;
    sub_149CCE0(v7, v24);
    v8 = v18;
  }
  v6(a2, v8, 0);
  sub_1BFB9A0(v22, *(_QWORD *)(*(_QWORD *)(a1 + 1080) + 8LL), *(_QWORD *)(*(_QWORD *)(a1 + 1080) + 16LL), v21 == 23);
  HIDWORD(v22[0]) = *(_DWORD *)(*(_QWORD *)(a1 + 1080) + 24LL);
  v9 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a2 + 16LL);
  v10 = sub_22077B0(208);
  v11 = v10;
  if ( v10 )
    sub_1BFB520(v10, v22);
  v9(a2, v11, 0);
  if ( v30 )
    j_j___libc_free_0(v30, v31 - v30);
  if ( v28 )
    j_j___libc_free_0(v28, v29 - v28);
  if ( v27 )
  {
    v12 = v26;
    v13 = v26 + 40LL * v27;
    do
    {
      while ( 1 )
      {
        if ( *(_DWORD *)v12 <= 0xFFFFFFFD )
        {
          v14 = *(_QWORD *)(v12 + 8);
          if ( v14 != v12 + 24 )
            break;
        }
        v12 += 40;
        if ( v13 == v12 )
          goto LABEL_21;
      }
      v15 = *(_QWORD *)(v12 + 24);
      v12 += 40;
      j_j___libc_free_0(v14, v15 + 1);
    }
    while ( v13 != v12 );
  }
LABEL_21:
  j___libc_free_0(v26);
  result = v20;
  if ( v19 != v20 )
    return (_QWORD *)j_j___libc_free_0(v19, v20[0] + 1LL);
  return result;
}

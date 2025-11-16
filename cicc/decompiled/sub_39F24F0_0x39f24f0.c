// Function: sub_39F24F0
// Address: 0x39f24f0
//
__int64 __fastcall sub_39F24F0(__int64 a1, _QWORD **a2, _QWORD **a3, __int64 *a4, char a5, char a6, char a7)
{
  _QWORD *v7; // r12
  _QWORD *v8; // r13
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r15
  __int64 v14; // r14
  __int64 v16; // [rsp+10h] [rbp-60h]
  _QWORD *v19; // [rsp+28h] [rbp-48h] BYREF
  _QWORD *v20; // [rsp+30h] [rbp-40h] BYREF
  __int64 v21[7]; // [rsp+38h] [rbp-38h] BYREF

  v7 = *a2;
  *a2 = 0;
  v8 = *a3;
  *a3 = 0;
  v9 = *a4;
  *a4 = 0;
  v16 = v9;
  v10 = sub_22077B0(0x168u);
  v12 = v16;
  v13 = v10;
  if ( v10 )
  {
    v21[0] = v16;
    v20 = v8;
    v14 = v10;
    v19 = v7;
    sub_38D3FD0(v10, a1, (__int64 *)&v19, (__int64 *)&v20, v21);
    if ( v19 )
      (*(void (__fastcall **)(_QWORD *))(*v19 + 8LL))(v19);
    if ( v20 )
      (*(void (__fastcall **)(_QWORD *))(*v20 + 8LL))(v20);
    if ( v21[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v21[0] + 8LL))(v21[0]);
    *(_BYTE *)(v13 + 322) = 0;
    *(_QWORD *)v13 = off_4A40FA0;
    *(_QWORD *)(v13 + 328) = 0;
    *(_BYTE *)(v13 + 320) = a7;
    *(_QWORD *)(v13 + 336) = 0;
    *(_BYTE *)(v13 + 321) = a6;
    *(_QWORD *)(v13 + 344) = 0;
    *(_DWORD *)(v13 + 352) = 0;
  }
  else
  {
    v14 = 0;
    if ( v16 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
    if ( v8 )
      (*(void (__fastcall **)(_QWORD *, _QWORD **, _QWORD, __int64, __int64))(*v8 + 8LL))(v8, a2, *v8, v11, v12);
    if ( v7 )
      (*(void (__fastcall **)(_QWORD *, _QWORD **, _QWORD, __int64, __int64))(*v7 + 8LL))(v7, a2, *v7, v11, v12);
  }
  sub_38DDF50(v14, *(_QWORD *)(a1 + 32) + 696LL);
  if ( a5 )
    *(_BYTE *)(*(_QWORD *)(v13 + 264) + 484LL) |= 1u;
  return v13;
}

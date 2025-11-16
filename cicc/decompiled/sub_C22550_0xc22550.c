// Function: sub_C22550
// Address: 0xc22550
//
__int64 __fastcall sub_C22550(__int64 a1, _QWORD *a2)
{
  __int64 *v3; // rdx
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 (__fastcall ***v7)(); // r13
  __int64 v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rdx
  char *(*v11)(); // rcx
  char *v12; // rax
  _QWORD v13[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v14; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v15; // [rsp+20h] [rbp-80h] BYREF
  __int16 v16; // [rsp+40h] [rbp-60h]
  _QWORD v17[4]; // [rsp+50h] [rbp-50h] BYREF
  int v18; // [rsp+70h] [rbp-30h]
  _QWORD *v19; // [rsp+78h] [rbp-28h]

  v3 = (__int64 *)a2[26];
  v4 = v3 + 1;
  if ( (unsigned __int64)(v3 + 1) > a2[27] )
  {
    v7 = sub_C1AFD0();
    ((void (__fastcall *)(_QWORD *, __int64 (__fastcall ***)(), __int64))(*v7)[4])(v13, v7, 4);
    v8 = a2[9];
    v15 = v13;
    v16 = 260;
    v9 = a2[8];
    v10 = 14;
    v11 = *(char *(**)())(*(_QWORD *)v8 + 16LL);
    v12 = "Unknown buffer";
    if ( v11 != sub_C1E8B0 )
      v12 = (char *)((__int64 (__fastcall *)(__int64, __int64 (__fastcall ***)(), __int64))v11)(v8, v7, 14);
    v17[2] = v12;
    v19 = &v15;
    v17[1] = 12;
    v17[0] = &unk_49D9C78;
    v17[3] = v10;
    v18 = 0;
    sub_B6EB20(v9, (__int64)v17);
    if ( (__int64 *)v13[0] != &v14 )
      j_j___libc_free_0(v13[0], v14 + 1);
    *(_QWORD *)(a1 + 8) = v7;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 4;
    return a1;
  }
  else
  {
    v5 = *v3;
    a2[26] = v4;
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v5;
    return a1;
  }
}

// Function: sub_C20B00
// Address: 0xc20b00
//
__int64 __fastcall sub_C20B00(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  size_t v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 (__fastcall ***v8)(); // r13
  __int64 v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rdx
  char *(*v12)(); // rcx
  char *v13; // rax
  _QWORD v14[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v15; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v16; // [rsp+20h] [rbp-80h] BYREF
  __int16 v17; // [rsp+40h] [rbp-60h]
  _QWORD v18[4]; // [rsp+50h] [rbp-50h] BYREF
  int v19; // [rsp+70h] [rbp-30h]
  _QWORD *v20; // [rsp+78h] [rbp-28h]

  v3 = *(_QWORD *)(a2 + 208);
  if ( v3 )
  {
    v4 = strlen(*(const char **)(a2 + 208));
    v5 = v4 + 1;
  }
  else
  {
    v5 = 1;
    v4 = 0;
  }
  v6 = v3 + v5;
  if ( v6 > *(_QWORD *)(a2 + 216) )
  {
    v8 = sub_C1AFD0();
    ((void (__fastcall *)(_QWORD *, __int64 (__fastcall ***)(), __int64))(*v8)[4])(v14, v8, 4);
    v9 = *(_QWORD *)(a2 + 72);
    v16 = v14;
    v17 = 260;
    v10 = *(_QWORD *)(a2 + 64);
    v11 = 14;
    v12 = *(char *(**)())(*(_QWORD *)v9 + 16LL);
    v13 = "Unknown buffer";
    if ( v12 != sub_C1E8B0 )
      v13 = (char *)((__int64 (__fastcall *)(__int64, __int64 (__fastcall ***)(), __int64))v12)(v9, v8, 14);
    v18[2] = v13;
    v20 = &v16;
    v18[1] = 12;
    v18[0] = &unk_49D9C78;
    v18[3] = v11;
    v19 = 0;
    sub_B6EB20(v10, (__int64)v18);
    if ( (__int64 *)v14[0] != &v15 )
      j_j___libc_free_0(v14[0], v15 + 1);
    *(_QWORD *)(a1 + 8) = v8;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 4;
    return a1;
  }
  else
  {
    *(_QWORD *)(a2 + 208) = v6;
    *(_QWORD *)a1 = v3;
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)(a1 + 8) = v4;
    return a1;
  }
}

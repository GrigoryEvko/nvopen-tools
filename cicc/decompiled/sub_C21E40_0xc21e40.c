// Function: sub_C21E40
// Address: 0xc21e40
//
__int64 __fastcall sub_C21E40(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // ecx
  __int64 v4; // rdi
  char *v6; // r9
  char *v7; // rax
  char v8; // dl
  __int64 v9; // rsi
  __int64 v10; // rsi
  char *v11; // rax
  __int64 (__fastcall ***v13)(); // r13
  __int64 v14; // rdi
  __int64 v15; // r14
  __int64 v16; // rdx
  char *(*v17)(); // rcx
  char *v18; // rax
  _QWORD v19[2]; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v20; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v21; // [rsp+20h] [rbp-80h] BYREF
  __int16 v22; // [rsp+40h] [rbp-60h]
  _QWORD v23[4]; // [rsp+50h] [rbp-50h] BYREF
  int v24; // [rsp+70h] [rbp-30h]
  _QWORD *v25; // [rsp+78h] [rbp-28h]

  v2 = 0;
  v4 = 0;
  v6 = (char *)a2[26];
  v7 = v6;
  while ( v7 )
  {
    v8 = *v7;
    v9 = *v7 & 0x7F;
    if ( v2 > 0x3E )
    {
      if ( v2 == 63 )
      {
        if ( v9 != (v8 & 1) )
          break;
      }
      else if ( (*v7 & 0x7F) != 0 )
      {
        break;
      }
    }
    v10 = v9 << v2;
    ++v7;
    v2 += 7;
    v4 += v10;
    if ( v8 >= 0 )
      goto LABEL_6;
  }
  v4 = 0;
LABEL_6:
  v11 = &v6[(_DWORD)v7 - (_DWORD)v6];
  if ( (unsigned __int64)v11 > a2[27] )
  {
    v13 = sub_C1AFD0();
    ((void (__fastcall *)(_QWORD *, __int64 (__fastcall ***)(), __int64))(*v13)[4])(v19, v13, 4);
    v14 = a2[9];
    v21 = v19;
    v22 = 260;
    v15 = a2[8];
    v16 = 14;
    v17 = *(char *(**)())(*(_QWORD *)v14 + 16LL);
    v18 = "Unknown buffer";
    if ( v17 != sub_C1E8B0 )
      v18 = (char *)((__int64 (__fastcall *)(__int64, __int64 (__fastcall ***)(), __int64))v17)(v14, v13, 14);
    v23[2] = v18;
    v25 = &v21;
    v23[1] = 12;
    v23[0] = &unk_49D9C78;
    v23[3] = v16;
    v24 = 0;
    sub_B6EB20(v15, (__int64)v23);
    if ( (__int64 *)v19[0] != &v20 )
      j_j___libc_free_0(v19[0], v20 + 1);
    *(_QWORD *)(a1 + 8) = v13;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 4;
    return a1;
  }
  else
  {
    a2[26] = v11;
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_QWORD *)a1 = v4;
    return a1;
  }
}

// Function: sub_C22200
// Address: 0xc22200
//
__int64 __fastcall sub_C22200(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // ecx
  unsigned __int64 v4; // rdi
  char *v6; // r9
  char *v7; // rax
  char v8; // dl
  __int64 v9; // rsi
  __int64 v10; // rsi
  int v11; // eax
  char *v12; // rax
  __int64 (__fastcall ***v14)(); // r13
  __int64 v15; // rdi
  __int64 v16; // r14
  __int64 v17; // rdx
  char *(*v18)(); // rcx
  char *v19; // rax
  __int64 (__fastcall ***v20)(); // r13
  __int64 v21; // rdi
  __int64 v22; // r14
  __int64 v23; // rdx
  char *(*v24)(); // rcx
  char *v25; // rax
  _QWORD v26[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v27[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v28[4]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v29; // [rsp+40h] [rbp-60h]
  void *v30; // [rsp+50h] [rbp-50h] BYREF
  __int64 v31; // [rsp+58h] [rbp-48h]
  char *v32; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]
  int v34; // [rsp+70h] [rbp-30h]
  _QWORD *v35; // [rsp+78h] [rbp-28h]

  v2 = 0;
  v4 = 0;
  v6 = (char *)a2[26];
  v7 = v6;
  do
  {
    if ( !v7 )
    {
LABEL_5:
      v11 = (_DWORD)v7 - (_DWORD)v6;
      LODWORD(v4) = 0;
      goto LABEL_6;
    }
    v8 = *v7;
    v9 = *v7 & 0x7F;
    if ( v2 > 0x3E )
    {
      if ( v2 == 63 )
      {
        if ( v9 != (v8 & 1) )
          goto LABEL_5;
      }
      else if ( (*v7 & 0x7F) != 0 )
      {
        goto LABEL_5;
      }
    }
    v10 = v9 << v2;
    ++v7;
    v2 += 7;
    v4 += v10;
  }
  while ( v8 < 0 );
  if ( v4 <= 0xFFFFFFFF )
  {
    v11 = (_DWORD)v7 - (_DWORD)v6;
LABEL_6:
    v12 = &v6[v11];
    if ( (unsigned __int64)v12 > a2[27] )
    {
      v14 = sub_C1AFD0();
      ((void (__fastcall *)(_QWORD *, __int64 (__fastcall ***)(), __int64))(*v14)[4])(v26, v14, 4);
      v15 = a2[9];
      v28[0] = v26;
      v29 = 260;
      v16 = a2[8];
      v17 = 14;
      v18 = *(char *(**)())(*(_QWORD *)v15 + 16LL);
      v19 = "Unknown buffer";
      if ( v18 != sub_C1E8B0 )
        v19 = (char *)((__int64 (__fastcall *)(__int64, __int64 (__fastcall ***)(), __int64))v18)(v15, v14, 14);
      v32 = v19;
      v35 = v28;
      v31 = 12;
      v30 = &unk_49D9C78;
      v33 = v17;
      v34 = 0;
      sub_B6EB20(v16, (__int64)&v30);
      if ( (_QWORD *)v26[0] != v27 )
        j_j___libc_free_0(v26[0], v27[0] + 1LL);
      *(_QWORD *)(a1 + 8) = v14;
      *(_BYTE *)(a1 + 16) |= 1u;
      *(_DWORD *)a1 = 4;
      return a1;
    }
    else
    {
      a2[26] = v12;
      *(_BYTE *)(a1 + 16) &= ~1u;
      *(_DWORD *)a1 = v4;
      return a1;
    }
  }
  v20 = sub_C1AFD0();
  ((void (__fastcall *)(_QWORD *, __int64 (__fastcall ***)(), __int64))(*v20)[4])(v26, v20, 5);
  v21 = a2[9];
  v28[0] = v26;
  v29 = 260;
  v22 = a2[8];
  v23 = 14;
  v24 = *(char *(**)())(*(_QWORD *)v21 + 16LL);
  v25 = "Unknown buffer";
  if ( v24 != sub_C1E8B0 )
    v25 = (char *)((__int64 (__fastcall *)(__int64, __int64 (__fastcall ***)(), __int64))v24)(v21, v20, 14);
  v32 = v25;
  v35 = v28;
  v31 = 12;
  v30 = &unk_49D9C78;
  v33 = v23;
  v34 = 0;
  sub_B6EB20(v22, (__int64)&v30);
  if ( (_QWORD *)v26[0] != v27 )
    j_j___libc_free_0(v26[0], v27[0] + 1LL);
  *(_QWORD *)(a1 + 8) = v20;
  *(_BYTE *)(a1 + 16) |= 1u;
  *(_DWORD *)a1 = 5;
  return a1;
}

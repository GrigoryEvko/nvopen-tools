// Function: sub_3940120
// Address: 0x3940120
//
__int64 __fastcall sub_3940120(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rcx
  unsigned __int64 v3; // r14
  _BYTE *v6; // r9
  __int64 v7; // rdx
  _BYTE *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // rbx
  unsigned __int64 v14; // r9
  unsigned int v15; // r14d
  __int64 v16; // rbx
  __int64 v17; // rdi
  __int64 v18; // r15
  __int64 v19; // rdx
  char *(*v20)(); // rcx
  char *v21; // rax
  unsigned __int64 *v23; // [rsp+0h] [rbp-A0h] BYREF
  __int16 v24; // [rsp+10h] [rbp-90h]
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v27[4]; // [rsp+40h] [rbp-60h] BYREF
  int v28; // [rsp+60h] [rbp-40h]
  unsigned __int64 **v29; // [rsp+68h] [rbp-38h]

  v2 = 0;
  v3 = 0;
  v6 = (_BYTE *)a2[9];
  v7 = (unsigned __int8)*v6;
  v8 = v6;
  v9 = *v6 & 0x7F;
  while ( 1 )
  {
    v10 = v9 << v2;
    v11 = (unsigned __int64)(v9 << v2) >> v2;
    if ( v9 != v11 )
    {
LABEL_5:
      v12 = (_DWORD)v8 - (_DWORD)v6;
      LODWORD(v3) = 0;
      goto LABEL_6;
    }
    v3 += v10;
    v2 = (unsigned int)(v2 + 7);
    ++v8;
    if ( (v7 & 0x80u) == 0LL )
      break;
    v7 = (unsigned __int8)*v8;
    v9 = *v8 & 0x7F;
    if ( (_DWORD)v2 == 70 )
      goto LABEL_5;
  }
  v7 = 0xFFFFFFFFLL;
  if ( v3 > 0xFFFFFFFF )
  {
    v15 = 5;
    v16 = sub_393D180(v10, v9, 0xFFFFFFFFLL, v2, v11, (__int64)v6);
    goto LABEL_8;
  }
  v12 = (_DWORD)v8 - (_DWORD)v6;
LABEL_6:
  v13 = v12;
  v14 = (unsigned __int64)&v6[v12];
  if ( a2[10] >= v14 )
  {
    sub_393D180(v10, v9, v7, v2, v11, v14);
    a2[9] += v13;
    *(_BYTE *)(a1 + 16) &= ~1u;
    *(_DWORD *)a1 = v3;
    return a1;
  }
  v15 = 4;
  v16 = sub_393D180(v10, v9, v7, v2, v11, v14);
LABEL_8:
  (*(void (__fastcall **)(unsigned __int64 *, __int64, _QWORD))(*(_QWORD *)v16 + 32LL))(v25, v16, v15);
  v17 = a2[6];
  v23 = v25;
  v24 = 260;
  v18 = a2[5];
  v19 = 14;
  v20 = *(char *(**)())(*(_QWORD *)v17 + 16LL);
  v21 = "Unknown buffer";
  if ( v20 != sub_12BCB10 )
    v21 = (char *)((__int64 (__fastcall *)(__int64, __int64, __int64))v20)(v17, v16, 14);
  v27[2] = v21;
  v27[1] = 7;
  v29 = &v23;
  v27[0] = &unk_49ECF18;
  v27[3] = v19;
  v28 = 0;
  sub_16027F0(v18, (__int64)v27);
  if ( (__int64 *)v25[0] != &v26 )
    j_j___libc_free_0(v25[0]);
  *(_BYTE *)(a1 + 16) |= 1u;
  *(_DWORD *)a1 = v15;
  *(_QWORD *)(a1 + 8) = v16;
  return a1;
}

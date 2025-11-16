// Function: sub_393FF90
// Address: 0x393ff90
//
__int64 __fastcall sub_393FF90(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r14
  _BYTE *v6; // r9
  __int64 v7; // rdx
  _BYTE *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // r8
  unsigned int v12; // eax
  __int64 v13; // rbx
  unsigned __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // rdi
  __int64 v17; // r14
  __int64 v18; // rdx
  char *(*v19)(); // rcx
  char *v20; // rax
  unsigned __int64 *v22; // [rsp+0h] [rbp-90h] BYREF
  __int16 v23; // [rsp+10h] [rbp-80h]
  unsigned __int64 v24[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  _QWORD v26[4]; // [rsp+40h] [rbp-50h] BYREF
  int v27; // [rsp+60h] [rbp-30h]
  unsigned __int64 **v28; // [rsp+68h] [rbp-28h]

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
      v3 = 0;
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
  v12 = (_DWORD)v8 - (_DWORD)v6;
LABEL_6:
  v13 = v12;
  v14 = (unsigned __int64)&v6[v12];
  if ( a2[10] >= v14 )
  {
    sub_393D180(v10, v9, v7, v2, v11, v14);
    a2[9] += v13;
    *(_QWORD *)a1 = v3;
    *(_BYTE *)(a1 + 16) &= ~1u;
    return a1;
  }
  else
  {
    v15 = sub_393D180(v10, v9, v7, v2, v11, v14);
    (*(void (__fastcall **)(unsigned __int64 *, __int64, __int64))(*(_QWORD *)v15 + 32LL))(v24, v15, 4);
    v16 = a2[6];
    v22 = v24;
    v23 = 260;
    v17 = a2[5];
    v18 = 14;
    v19 = *(char *(**)())(*(_QWORD *)v16 + 16LL);
    v20 = "Unknown buffer";
    if ( v19 != sub_12BCB10 )
      v20 = (char *)((__int64 (__fastcall *)(__int64, __int64, __int64))v19)(v16, v15, 14);
    v26[2] = v20;
    v26[1] = 7;
    v28 = &v22;
    v26[0] = &unk_49ECF18;
    v26[3] = v18;
    v27 = 0;
    sub_16027F0(v17, (__int64)v26);
    if ( (__int64 *)v24[0] != &v25 )
      j_j___libc_free_0(v24[0]);
    *(_QWORD *)(a1 + 8) = v15;
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = 4;
    return a1;
  }
}

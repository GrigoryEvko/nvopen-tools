// Function: sub_16BEB10
// Address: 0x16beb10
//
__int64 __fastcall sub_16BEB10(__int64 a1, __int64 a2, _DWORD *a3)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // r15
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rdi
  _BYTE *v12; // rax
  char *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  const char *v20[2]; // [rsp+0h] [rbp-E0h] BYREF
  __int64 v21; // [rsp+10h] [rbp-D0h] BYREF
  char *v22; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v23; // [rsp+28h] [rbp-B8h]
  _BYTE v24[176]; // [rsp+30h] [rbp-B0h] BYREF

  *a3 = -1;
  v22 = v24;
  v23 = 0x8000000000LL;
  v4 = sub_16C64B0(a2, "dot", 3, a3, &v22);
  if ( v4 )
  {
    v6 = v5;
    v7 = v4;
    v8 = sub_16E8CB0(a2, "dot", v5);
    v9 = *(_QWORD *)(v8 + 24);
    v10 = v8;
    if ( (unsigned __int64)(*(_QWORD *)(v8 + 16) - v9) <= 6 )
    {
      v10 = sub_16E7EE0(v8, "Error: ", 7);
    }
    else
    {
      *(_DWORD *)v9 = 1869771333;
      *(_WORD *)(v9 + 4) = 14962;
      *(_BYTE *)(v9 + 6) = 32;
      *(_QWORD *)(v8 + 24) += 7LL;
    }
    (*(void (__fastcall **)(const char **, __int64, _QWORD))(*(_QWORD *)v6 + 32LL))(v20, v6, v7);
    v11 = sub_16E7EE0(v10, v20[0], v20[1]);
    v12 = *(_BYTE **)(v11 + 24);
    if ( *(_BYTE **)(v11 + 16) == v12 )
    {
      sub_16E7EE0(v11, "\n", 1);
    }
    else
    {
      *v12 = 10;
      ++*(_QWORD *)(v11 + 24);
    }
    if ( (__int64 *)v20[0] != &v21 )
      j_j___libc_free_0(v20[0], v21 + 1);
    *(_QWORD *)a1 = a1 + 16;
    sub_16BE2B0((__int64 *)a1, byte_3F871B3, (__int64)byte_3F871B3);
    v13 = v22;
  }
  else
  {
    v14 = sub_16E8CB0(a2, "dot", v5);
    v15 = *(_QWORD *)(v14 + 24);
    v16 = v14;
    if ( (unsigned __int64)(*(_QWORD *)(v14 + 16) - v15) <= 8 )
    {
      v16 = sub_16E7EE0(v14, "Writing '", 9);
    }
    else
    {
      *(_BYTE *)(v15 + 8) = 39;
      *(_QWORD *)v15 = 0x20676E6974697257LL;
      *(_QWORD *)(v14 + 24) += 9LL;
    }
    v17 = sub_16E7EE0(v16, v22, (unsigned int)v23);
    v18 = *(_QWORD *)(v17 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(v17 + 16) - v18) <= 4 )
    {
      sub_16E7EE0(v17, "'... ", 5);
    }
    else
    {
      *(_DWORD *)v18 = 774778407;
      *(_BYTE *)(v18 + 4) = 32;
      *(_QWORD *)(v17 + 24) += 5LL;
    }
    v13 = v22;
    *(_QWORD *)a1 = a1 + 16;
    if ( v13 )
    {
      sub_16BE2B0((__int64 *)a1, v13, (__int64)&v13[(unsigned int)v23]);
      v13 = v22;
    }
    else
    {
      *(_QWORD *)(a1 + 8) = 0;
      *(_BYTE *)(a1 + 16) = 0;
    }
  }
  if ( v13 != v24 )
    _libc_free((unsigned __int64)v13);
  return a1;
}

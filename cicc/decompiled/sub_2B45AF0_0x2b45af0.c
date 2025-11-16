// Function: sub_2B45AF0
// Address: 0x2b45af0
//
__int64 __fastcall sub_2B45AF0(__int64 a1, unsigned int a2)
{
  __int64 v2; // r8
  unsigned __int64 v5; // rax
  __int64 v6; // rax
  char *v7; // r12
  __int64 v8; // r13
  __int64 *v9; // r15
  _QWORD *v10; // rax
  __int64 v11; // rax
  int v12; // r9d
  __int64 v13; // rdi
  int v14; // eax
  __int64 v15; // rax
  __int64 v16; // r9
  int v17; // edx
  bool v18; // sf
  bool v19; // of
  __int64 v20; // rsi
  __int64 v21; // rax
  int v22; // ecx
  bool v23; // cl
  unsigned int v25; // [rsp+0h] [rbp-A0h]
  int v26; // [rsp+4h] [rbp-9Ch]
  __int64 v27; // [rsp+8h] [rbp-98h]
  __int64 v28; // [rsp+10h] [rbp-90h] BYREF
  int v29; // [rsp+18h] [rbp-88h]
  __int64 v30; // [rsp+20h] [rbp-80h]
  int v31; // [rsp+28h] [rbp-78h]
  char *v32; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v33; // [rsp+38h] [rbp-68h]
  char v34; // [rsp+40h] [rbp-60h] BYREF

  LODWORD(v2) = 0;
  if ( a2 )
  {
    LODWORD(v2) = 1;
    if ( a2 != 1 )
    {
      _BitScanReverse64(&v5, a2 - 1LL);
      v2 = 1LL << (64 - ((unsigned __int8)v5 ^ 0x3Fu));
    }
  }
  v25 = v2;
  sub_2B1F480(
    (__int64)&v32,
    **(unsigned __int8 ***)a1,
    **(_DWORD **)(a1 + 8),
    **(_DWORD **)(a1 + 16),
    v2,
    *(_QWORD *)(*(_QWORD *)(a1 + 24) + 3296LL));
  v6 = *(_QWORD *)(a1 + 24);
  v7 = v32;
  v8 = v33;
  v9 = *(__int64 **)(v6 + 3304);
  v27 = *(_QWORD *)(v6 + 3296);
  v26 = **(_DWORD **)(a1 + 16);
  v10 = (_QWORD *)sub_BD5C60(**(_QWORD **)a1);
  v11 = sub_BCCE00(v10, v25);
  v12 = v26;
  v13 = v11;
  v14 = *(unsigned __int8 *)(v11 + 8);
  if ( (_BYTE)v14 == 17 )
  {
    v12 = *(_DWORD *)(v13 + 32) * v26;
LABEL_6:
    v13 = **(_QWORD **)(v13 + 16);
    goto LABEL_7;
  }
  if ( (unsigned int)(v14 - 17) <= 1 )
    goto LABEL_6;
LABEL_7:
  v15 = sub_BCDA70((__int64 *)v13, v12);
  sub_2B45470((unsigned int *)&v28, **(_QWORD **)a1, v15, v27, v9, v16, v7, v8);
  v17 = v29;
  v19 = __OFSUB__(v31, v29);
  v18 = v31 - v29 < 0;
  if ( v31 == v29 )
  {
    v19 = __OFSUB__(v30, v28);
    v18 = v30 - v28 < 0;
  }
  if ( v18 != v19 )
  {
    v20 = v30;
    v21 = *(_QWORD *)(a1 + 32);
    v17 = v31;
    v22 = *(_DWORD *)(v21 + 8);
    if ( v22 != v31 )
      goto LABEL_11;
  }
  else
  {
    v20 = v28;
    v21 = *(_QWORD *)(a1 + 32);
    v22 = *(_DWORD *)(v21 + 8);
    if ( v22 != v29 )
    {
LABEL_11:
      v23 = v17 < v22;
      goto LABEL_12;
    }
  }
  v23 = v20 < *(_QWORD *)v21;
LABEL_12:
  if ( v23 )
  {
    *(_QWORD *)v21 = v20;
    *(_DWORD *)(v21 + 8) = v17;
    **(_DWORD **)(a1 + 40) = a2;
  }
  if ( v32 != &v34 )
    _libc_free((unsigned __int64)v32);
  return 0;
}

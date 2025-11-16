// Function: sub_10235B0
// Address: 0x10235b0
//
__int64 __fastcall sub_10235B0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  __int64 *v7; // rdi
  __int64 v9; // r8
  bool v10; // zf
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  _BYTE *v14; // r14
  __int64 v15; // r15
  char v16; // al
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // rsi
  _QWORD *v20; // rax
  _QWORD *v21; // rcx
  __int64 *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 *v25; // rax
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // [rsp+0h] [rbp-90h]
  _QWORD *v32; // [rsp+0h] [rbp-90h]
  __int64 v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+8h] [rbp-88h]
  _QWORD v35[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v36; // [rsp+20h] [rbp-70h]
  int v37; // [rsp+28h] [rbp-68h]
  __int64 v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+38h] [rbp-58h]
  char *v40[2]; // [rsp+40h] [rbp-50h] BYREF
  char v41; // [rsp+50h] [rbp-40h] BYREF

  if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(a1 + 40) || (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v7 = *(__int64 **)(a1 - 8);
  v9 = a2 + 56;
  v10 = *(_BYTE *)(a2 + 84) == 0;
  v11 = v7[4 * *(unsigned int *)(a1 + 72)];
  if ( !v10 )
  {
    v12 = *(_QWORD **)(a2 + 64);
    v13 = &v12[*(unsigned int *)(a2 + 76)];
    if ( v12 == v13 )
      goto LABEL_25;
    while ( v11 != *v12 )
    {
      if ( v13 == ++v12 )
        goto LABEL_25;
    }
LABEL_9:
    v14 = (_BYTE *)*v7;
    v15 = v7[4];
    goto LABEL_10;
  }
  v31 = a3;
  v33 = v9;
  v22 = sub_C8CA60(v9, v11);
  v7 = *(__int64 **)(a1 - 8);
  v9 = v33;
  a3 = v31;
  if ( v22 )
    goto LABEL_9;
LABEL_25:
  v14 = (_BYTE *)v7[4];
  v15 = *v7;
LABEL_10:
  v16 = *v14;
  if ( (unsigned __int8)(*v14 - 42) > 0x11u )
    return 0;
  if ( v16 == 43 )
  {
    v18 = *((_QWORD *)v14 - 8);
    v23 = *((_QWORD *)v14 - 4);
    if ( v18 && a1 == v18 )
    {
      if ( !v23 )
        return 0;
      v18 = *((_QWORD *)v14 - 4);
      goto LABEL_17;
    }
    if ( a1 != v23 || v23 == 0 )
      return 0;
  }
  else
  {
    if ( v16 != 45 )
      return 0;
    v17 = *((_QWORD *)v14 - 8);
    if ( !v17 || a1 != v17 )
      return 0;
    v18 = *((_QWORD *)v14 - 4);
  }
  if ( !v18 )
    return 0;
LABEL_17:
  if ( *(_BYTE *)v18 > 0x1Cu )
  {
    v19 = *(_QWORD *)(v18 + 40);
    if ( *(_BYTE *)(a2 + 84) )
    {
      v20 = *(_QWORD **)(a2 + 64);
      v21 = &v20[*(unsigned int *)(a2 + 76)];
      if ( v20 == v21 )
        goto LABEL_31;
      while ( v19 != *v20 )
      {
        if ( v21 == ++v20 )
          goto LABEL_31;
      }
      return 0;
    }
    v32 = a3;
    v34 = v18;
    v24 = sub_C8CA60(v9, v19);
    v18 = v34;
    a3 = v32;
    if ( v24 )
      return 0;
  }
LABEL_31:
  v25 = sub_DA3860(a3, v18);
  sub_1023480((__int64)v35, v15, 3, (__int64)v25, (__int64)v14, 0);
  v29 = *(_QWORD *)(a4 + 16);
  v30 = v36;
  if ( v29 != v36 )
  {
    if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
    {
      sub_BD60C0((_QWORD *)a4);
      v30 = v36;
    }
    *(_QWORD *)(a4 + 16) = v30;
    LOBYTE(v26) = v30 != -4096;
    LOBYTE(v29) = v30 != 0;
    if ( ((v30 != 0) & (unsigned __int8)v26) != 0 && v30 != -8192 )
      sub_BD6050((unsigned __int64 *)a4, v35[0] & 0xFFFFFFFFFFFFFFF8LL);
  }
  *(_DWORD *)(a4 + 24) = v37;
  *(_QWORD *)(a4 + 32) = v38;
  *(_QWORD *)(a4 + 40) = v39;
  sub_1021AD0(a4 + 48, v40, v29, v26, v27, v28);
  if ( v40[0] != &v41 )
    _libc_free(v40[0], v40);
  if ( v36 != -4096 && v36 != 0 && v36 != -8192 )
    sub_BD60C0(v35);
  return 1;
}

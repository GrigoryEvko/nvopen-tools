// Function: sub_159E800
// Address: 0x159e800
//
__int64 __fastcall sub_159E800(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v5; // r8
  __int64 ***v6; // r15
  int v7; // edx
  unsigned __int64 v8; // rdx
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 *v11; // r13
  __int64 *v12; // rbx
  int v13; // r11d
  char v14; // r14
  __int64 v15; // r15
  __int64 v16; // r12
  __int64 v17; // r12
  bool v19; // al
  _QWORD *v20; // rax
  __int64 v21; // [rsp+8h] [rbp-B8h]
  __int64 *v22; // [rsp+10h] [rbp-B0h]
  int v23; // [rsp+18h] [rbp-A8h]
  __int64 v24; // [rsp+18h] [rbp-A8h]
  __int64 v25; // [rsp+18h] [rbp-A8h]
  __int64 v26; // [rsp+20h] [rbp-A0h]
  int v27; // [rsp+20h] [rbp-A0h]
  __int64 v28; // [rsp+20h] [rbp-A0h]
  int v29; // [rsp+28h] [rbp-98h]
  __int64 v30; // [rsp+28h] [rbp-98h]
  __int64 v31; // [rsp+28h] [rbp-98h]
  unsigned int v32; // [rsp+38h] [rbp-88h]
  __int64 *v33; // [rsp+40h] [rbp-80h] BYREF
  __int64 v34; // [rsp+48h] [rbp-78h]
  _BYTE v35[112]; // [rsp+50h] [rbp-70h] BYREF

  v5 = a2;
  v6 = (__int64 ***)a1;
  v7 = *(_DWORD *)(a1 + 20);
  v33 = (__int64 *)v35;
  v8 = v7 & 0xFFFFFFF;
  v34 = 0x800000000LL;
  if ( v8 > 8 )
  {
    v31 = a2;
    a2 = (__int64)v35;
    v28 = a3;
    sub_16CD150(&v33, v35, v8, 8);
    a3 = v28;
    v5 = v31;
    v8 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  }
  v9 = 3 * v8;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) == 0 )
  {
    v10 = (__int64 *)(a1 - v9 * 8);
    v11 = (__int64 *)a1;
    if ( (__int64 *)a1 != v10 )
      goto LABEL_5;
LABEL_18:
    v32 = 0;
    v13 = 0;
    goto LABEL_19;
  }
  v10 = *(__int64 **)(a1 - 8);
  v11 = &v10[v9];
  if ( v11 == v10 )
    goto LABEL_18;
LABEL_5:
  v8 = (unsigned int)v34;
  v12 = v10;
  v13 = 0;
  v32 = 0;
  v14 = 1;
  v15 = v5;
  do
  {
    v16 = *v12;
    if ( *v12 == v15 )
    {
      ++v13;
      v16 = a3;
      v32 = -1431655765 * (v12 - v10);
      if ( HIDWORD(v34) > (unsigned int)v8 )
        goto LABEL_7;
    }
    else
    {
      v14 &= a3 == v16;
      if ( HIDWORD(v34) > (unsigned int)v8 )
        goto LABEL_7;
    }
    v21 = a3;
    v22 = v10;
    v23 = v13;
    sub_16CD150(&v33, v35, 0, 8);
    v8 = (unsigned int)v34;
    a3 = v21;
    v10 = v22;
    v13 = v23;
LABEL_7:
    a4 = v33;
    v12 += 3;
    v33[v8] = v16;
    a2 = (unsigned int)v34;
    v8 = (unsigned int)(v34 + 1);
    LODWORD(v34) = v34 + 1;
  }
  while ( v12 != v11 );
  v5 = v15;
  v6 = (__int64 ***)a1;
  if ( !v14 )
  {
LABEL_12:
    v24 = a3;
    v26 = v5;
    v29 = v13;
    v17 = sub_159A200(*v6, v33, v8, (__int64)a4);
    if ( !v17 )
    {
      v20 = (_QWORD *)sub_16498A0(v6);
      v17 = sub_159E1A0(*v20 + 1552LL, v33, (unsigned int)v34, (__int64 *)v6, v26, v24, v29, v32);
    }
    goto LABEL_14;
  }
LABEL_19:
  v25 = v5;
  v27 = v13;
  v30 = a3;
  v19 = sub_1593BB0(a3, a2, v8, (__int64)a4);
  a3 = v30;
  v13 = v27;
  v5 = v25;
  if ( v19 )
  {
    v17 = sub_1598F00(*v6);
  }
  else
  {
    if ( *(_BYTE *)(v30 + 16) != 9 )
    {
      v8 = (unsigned int)v34;
      goto LABEL_12;
    }
    v17 = sub_1599EF0(*v6);
  }
LABEL_14:
  if ( v33 != (__int64 *)v35 )
    _libc_free((unsigned __int64)v33);
  return v17;
}

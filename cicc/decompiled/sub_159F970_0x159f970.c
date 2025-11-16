// Function: sub_159F970
// Address: 0x159f970
//
__int64 __fastcall sub_159F970(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 ***v4; // r10
  _BYTE *v5; // r8
  unsigned __int64 v7; // rdx
  __int64 *v8; // r11
  __int64 *v9; // rax
  __int64 *v10; // r15
  __int64 *v11; // rbx
  __int64 v12; // rax
  int v13; // r9d
  char v14; // r14
  unsigned int v15; // r13d
  __int64 v16; // r12
  unsigned int v17; // ebx
  char v18; // r13
  _QWORD *v19; // rax
  __int64 v20; // r12
  bool v22; // al
  __int64 v23; // [rsp+8h] [rbp-B8h]
  _BYTE *v24; // [rsp+10h] [rbp-B0h]
  __int64 ***v25; // [rsp+18h] [rbp-A8h]
  __int64 *v26; // [rsp+20h] [rbp-A0h]
  _BYTE *v27; // [rsp+20h] [rbp-A0h]
  int v28; // [rsp+28h] [rbp-98h]
  __int64 v29; // [rsp+28h] [rbp-98h]
  _BYTE *v30; // [rsp+28h] [rbp-98h]
  int v31; // [rsp+30h] [rbp-90h]
  __int64 *v32; // [rsp+30h] [rbp-90h]
  __int64 ***v33; // [rsp+30h] [rbp-90h]
  __int64 *v34; // [rsp+38h] [rbp-88h]
  int v35; // [rsp+38h] [rbp-88h]
  __int64 *v36; // [rsp+40h] [rbp-80h] BYREF
  __int64 v37; // [rsp+48h] [rbp-78h]
  _BYTE v38[112]; // [rsp+50h] [rbp-70h] BYREF

  v4 = (__int64 ***)a1;
  v5 = a2;
  v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    v8 = *(__int64 **)(a1 - 8);
  else
    v8 = (__int64 *)(a1 - 24 * v7);
  v36 = (__int64 *)v38;
  v37 = 0x800000000LL;
  if ( v7 > 8 )
  {
    a2 = v38;
    v27 = v5;
    v32 = v8;
    sub_16CD150(&v36, v38, v7, 8);
    v4 = (__int64 ***)a1;
    v8 = v32;
    v5 = v27;
    v7 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
    v9 = &v32[3 * v7];
    if ( v9 != v32 )
      goto LABEL_5;
LABEL_18:
    v13 = 0;
    v17 = 0;
    goto LABEL_19;
  }
  v9 = &v8[3 * v7];
  if ( v9 == v8 )
    goto LABEL_18;
LABEL_5:
  a2 = v38;
  v7 = (unsigned int)v37;
  v10 = v9;
  v11 = v8;
  v12 = a3;
  v13 = 0;
  v14 = 1;
  v15 = 0;
  do
  {
    v16 = *v11;
    if ( (_BYTE *)*v11 == v5 )
    {
      ++v13;
      v16 = v12;
      v15 = -1431655765 * (v11 - v8);
      if ( (unsigned int)v7 < HIDWORD(v37) )
        goto LABEL_7;
    }
    else
    {
      v14 &= v12 == v16;
      if ( (unsigned int)v7 < HIDWORD(v37) )
        goto LABEL_7;
    }
    v23 = v12;
    v24 = v5;
    v25 = v4;
    v26 = v8;
    v28 = v13;
    sub_16CD150(&v36, v38, 0, 8);
    v7 = (unsigned int)v37;
    v12 = v23;
    v5 = v24;
    v4 = v25;
    v8 = v26;
    v13 = v28;
LABEL_7:
    v11 += 3;
    v36[v7] = v16;
    a4 = (unsigned int)v37;
    v7 = (unsigned int)(v37 + 1);
    LODWORD(v37) = v37 + 1;
  }
  while ( v10 != v11 );
  v17 = v15;
  v18 = v14;
  a3 = v12;
  if ( !v18 )
    goto LABEL_12;
LABEL_19:
  v30 = v5;
  v33 = v4;
  v35 = v13;
  v22 = sub_1593BB0(a3, (__int64)a2, v7, a4);
  v13 = v35;
  v4 = v33;
  v5 = v30;
  if ( v22 )
  {
    v20 = sub_1598F00(*v33);
  }
  else if ( *(_BYTE *)(a3 + 16) == 9 )
  {
    v20 = sub_1599EF0(*v33);
  }
  else
  {
LABEL_12:
    v29 = (__int64)v5;
    v31 = v13;
    v34 = (__int64 *)v4;
    v19 = (_QWORD *)sub_16498A0(v4);
    v20 = sub_159F310(*v19 + 1584LL, v36, (unsigned int)v37, v34, v29, a3, v31, v17);
  }
  if ( v36 != (__int64 *)v38 )
    _libc_free((unsigned __int64)v36);
  return v20;
}

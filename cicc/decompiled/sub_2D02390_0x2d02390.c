// Function: sub_2D02390
// Address: 0x2d02390
//
_QWORD *__fastcall sub_2D02390(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  char v7; // al
  unsigned __int64 v8; // rbx
  char v9; // r13
  unsigned __int64 v10; // rdi
  _QWORD *v11; // rsi
  _QWORD *v12; // rdx
  unsigned __int64 v14; // [rsp+0h] [rbp-130h] BYREF
  __int64 v15; // [rsp+10h] [rbp-120h] BYREF
  unsigned __int64 v16; // [rsp+18h] [rbp-118h]
  __int64 *v17; // [rsp+20h] [rbp-110h]
  __int64 *v18; // [rsp+28h] [rbp-108h]
  __int64 v19; // [rsp+30h] [rbp-100h]
  int v20; // [rsp+40h] [rbp-F0h] BYREF
  unsigned __int64 v21; // [rsp+48h] [rbp-E8h]
  int *v22; // [rsp+50h] [rbp-E0h]
  int *v23; // [rsp+58h] [rbp-D8h]
  __int64 v24; // [rsp+60h] [rbp-D0h]
  int v25; // [rsp+70h] [rbp-C0h] BYREF
  unsigned __int64 v26; // [rsp+78h] [rbp-B8h]
  int *v27; // [rsp+80h] [rbp-B0h]
  int *v28; // [rsp+88h] [rbp-A8h]
  __int64 v29; // [rsp+90h] [rbp-A0h]
  int v30; // [rsp+A0h] [rbp-90h] BYREF
  unsigned __int64 v31; // [rsp+A8h] [rbp-88h]
  int *v32; // [rsp+B0h] [rbp-80h]
  int *v33; // [rsp+B8h] [rbp-78h]
  __int64 v34; // [rsp+C0h] [rbp-70h]
  int v35; // [rsp+D0h] [rbp-60h] BYREF
  unsigned __int64 v36; // [rsp+D8h] [rbp-58h]
  int *v37; // [rsp+E0h] [rbp-50h]
  int *v38; // [rsp+E8h] [rbp-48h]
  __int64 v39; // [rsp+F0h] [rbp-40h]
  __int64 v40; // [rsp+F8h] [rbp-38h]
  __int64 v41; // [rsp+100h] [rbp-30h]

  v6 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v17 = &v15;
  v18 = &v15;
  v22 = &v20;
  v23 = &v20;
  v27 = &v25;
  v28 = &v25;
  v32 = &v30;
  v33 = &v30;
  v37 = &v35;
  v38 = &v35;
  v14 = 0x100000000LL;
  v15 = 0;
  v16 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v7 = sub_2D01410(&v14, a3, v6 + 8);
  v8 = v36;
  v9 = v7;
  while ( v8 )
  {
    sub_2CFFFC0(*(_QWORD *)(v8 + 24));
    v10 = v8;
    v8 = *(_QWORD *)(v8 + 16);
    j_j___libc_free_0(v10);
  }
  sub_2D00360(v31);
  sub_2D00190(v26);
  sub_2D00190(v21);
  sub_2D00360(v16);
  v11 = a1 + 4;
  v12 = a1 + 10;
  if ( v9 )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v11;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v12;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v11;
    a1[2] = 0x100000002LL;
    a1[6] = 0;
    a1[7] = v12;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  return a1;
}

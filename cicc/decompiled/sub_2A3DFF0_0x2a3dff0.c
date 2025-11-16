// Function: sub_2A3DFF0
// Address: 0x2a3dff0
//
void __fastcall sub_2A3DFF0(char *a1, __int64 a2, __int64 a3, __int64 a4)
{
  double v4; // xmm0_8
  double v5; // xmm1_8
  char v6; // al
  _BYTE *v7; // r13
  unsigned __int64 *v8; // rbx
  unsigned __int64 *v9; // r12
  unsigned __int64 v10; // rdi
  __int64 *v11; // r12
  __int64 v12; // [rsp+0h] [rbp-360h] BYREF
  __int64 v13; // [rsp+8h] [rbp-358h] BYREF
  double v14; // [rsp+18h] [rbp-348h] BYREF
  __int64 v15[2]; // [rsp+20h] [rbp-340h] BYREF
  __int64 *v16; // [rsp+30h] [rbp-330h]
  __int8 *v17; // [rsp+40h] [rbp-320h] BYREF
  size_t v18; // [rsp+48h] [rbp-318h]
  _BYTE v19[16]; // [rsp+50h] [rbp-310h] BYREF
  _QWORD *v20; // [rsp+60h] [rbp-300h] BYREF
  __int16 v21; // [rsp+80h] [rbp-2E0h]
  __int64 v22[2]; // [rsp+90h] [rbp-2D0h] BYREF
  __int64 v23; // [rsp+A0h] [rbp-2C0h]
  __int64 v24; // [rsp+A8h] [rbp-2B8h]
  __int64 v25; // [rsp+B0h] [rbp-2B0h]
  __int64 v26; // [rsp+B8h] [rbp-2A8h]
  __int8 **v27; // [rsp+C0h] [rbp-2A0h]
  _QWORD v28[4]; // [rsp+D0h] [rbp-290h] BYREF
  char v29; // [rsp+F0h] [rbp-270h]
  _QWORD v30[2]; // [rsp+F8h] [rbp-268h] BYREF
  _QWORD *v31; // [rsp+108h] [rbp-258h] BYREF
  _QWORD v32[4]; // [rsp+110h] [rbp-250h] BYREF
  char v33; // [rsp+130h] [rbp-230h]
  _QWORD v34[2]; // [rsp+138h] [rbp-228h] BYREF
  _QWORD v35[2]; // [rsp+148h] [rbp-218h] BYREF
  _QWORD v36[2]; // [rsp+158h] [rbp-208h] BYREF
  _QWORD v37[3]; // [rsp+168h] [rbp-1F8h] BYREF
  _QWORD v38[10]; // [rsp+180h] [rbp-1E0h] BYREF
  unsigned __int64 *v39; // [rsp+1D0h] [rbp-190h]
  unsigned int v40; // [rsp+1D8h] [rbp-188h]
  char v41; // [rsp+1E0h] [rbp-180h] BYREF

  v13 = a3;
  v12 = a4;
  if ( a3 < 0 )
    v4 = (double)(int)(a3 & 1 | ((unsigned __int64)a3 >> 1)) + (double)(int)(a3 & 1 | ((unsigned __int64)a3 >> 1));
  else
    v4 = (double)(int)a3;
  if ( v12 < 0 )
    v5 = (double)(int)(v12 & 1 | ((unsigned __int64)v12 >> 1)) + (double)(int)(v12 & 1 | ((unsigned __int64)v12 >> 1));
  else
    v5 = (double)(int)v12;
  v32[1] = 17;
  v32[2] = v37;
  v32[0] = "{0:P} ({1} / {2})";
  v30[1] = v32;
  v34[0] = &unk_49E6648;
  v35[0] = &unk_49E6648;
  v35[1] = &v13;
  v20 = v32;
  v6 = *a1;
  v34[1] = &v12;
  v36[0] = &unk_49E64E0;
  v36[1] = &v14;
  v37[0] = v36;
  v37[1] = v35;
  v37[2] = v34;
  v28[2] = &v31;
  v32[3] = 3;
  v33 = 1;
  v30[0] = &unk_4A22DB0;
  v31 = v30;
  v28[0] = "Potential performance regression from use of the llvm.expect intrinsic: Annotation was correct on {0} of prof"
           "iled executions.";
  v28[1] = 125;
  v28[3] = 1;
  v29 = 1;
  v21 = 263;
  v14 = v4 / v5;
  if ( v6 == 31 )
  {
    v7 = (_BYTE *)*((_QWORD *)a1 - 12);
    if ( *v7 <= 0x1Cu )
      v7 = a1;
  }
  else
  {
    if ( v6 == 32 )
    {
      v7 = (_BYTE *)**((_QWORD **)a1 - 1);
      if ( !v7 )
        BUG();
      if ( *v7 <= 0x1Cu )
        v7 = a1;
      if ( (_BYTE)qword_500AFE8 )
        goto LABEL_9;
      goto LABEL_31;
    }
    v7 = a1;
  }
  if ( (_BYTE)qword_500AFE8 )
  {
LABEL_9:
    sub_B17D60((__int64)v38, (__int64)v7, (__int64)&v20);
    sub_B6EB20(a2, (__int64)v38);
    goto LABEL_10;
  }
LABEL_31:
  if ( (unsigned __int8)sub_B6E930(a2) )
    goto LABEL_9;
LABEL_10:
  sub_1049690(v15, *(_QWORD *)(*((_QWORD *)a1 + 5) + 72LL));
  sub_B174A0((__int64)v38, (__int64)"misexpect", (__int64)"misexpect", 9, (__int64)v7);
  v17 = v19;
  v26 = 0x100000000LL;
  v22[0] = (__int64)&unk_49DD210;
  v27 = &v17;
  v18 = 0;
  v19[0] = 0;
  v22[1] = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  sub_CB5980((__int64)v22, 0, 0, 0);
  sub_CB6840((__int64)v22, (__int64)v28);
  if ( v25 != v23 )
    sub_CB5AE0(v22);
  v22[0] = (__int64)&unk_49DD210;
  sub_CB5840((__int64)v22);
  sub_B18290((__int64)v38, v17, v18);
  sub_1049740(v15, (__int64)v38);
  if ( v17 != v19 )
    j_j___libc_free_0((unsigned __int64)v17);
  v8 = v39;
  v38[0] = &unk_49D9D40;
  v9 = &v39[10 * v40];
  if ( v39 != v9 )
  {
    do
    {
      v9 -= 10;
      v10 = v9[4];
      if ( (unsigned __int64 *)v10 != v9 + 6 )
        j_j___libc_free_0(v10);
      if ( (unsigned __int64 *)*v9 != v9 + 2 )
        j_j___libc_free_0(*v9);
    }
    while ( v8 != v9 );
    v9 = v39;
  }
  if ( v9 != (unsigned __int64 *)&v41 )
    _libc_free((unsigned __int64)v9);
  v11 = v16;
  if ( v16 )
  {
    sub_FDC110(v16);
    j_j___libc_free_0((unsigned __int64)v11);
  }
}

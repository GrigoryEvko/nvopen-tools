// Function: sub_12EA530
// Address: 0x12ea530
//
__int64 *__fastcall sub_12EA530(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  char *v5; // rsi
  size_t v6; // rax
  __int64 v7; // rcx
  _QWORD *v8; // rbx
  _QWORD *v9; // r13
  __int64 v11; // rsi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 (__fastcall *v15)(__int64, _QWORD *, __int64, __int64, _QWORD *, __int64, __int64 *, _BYTE *, __int64 *, __int64, _QWORD); // rbx
  __int64 v16; // rax
  _QWORD *v17; // [rsp+0h] [rbp-1F0h]
  __int64 v18; // [rsp+8h] [rbp-1E8h]
  __int64 v19; // [rsp+10h] [rbp-1E0h]
  __int64 v20; // [rsp+18h] [rbp-1D8h]
  __int64 v22; // [rsp+28h] [rbp-1C8h]
  __int64 v23; // [rsp+28h] [rbp-1C8h]
  _BYTE v24[8]; // [rsp+30h] [rbp-1C0h] BYREF
  __int64 v25; // [rsp+38h] [rbp-1B8h] BYREF
  _QWORD v26[4]; // [rsp+40h] [rbp-1B0h] BYREF
  __int16 v27; // [rsp+60h] [rbp-190h]
  _QWORD v28[2]; // [rsp+70h] [rbp-180h] BYREF
  _QWORD v29[2]; // [rsp+80h] [rbp-170h] BYREF
  _QWORD *v30; // [rsp+90h] [rbp-160h]
  __int64 v31; // [rsp+98h] [rbp-158h]
  _QWORD v32[2]; // [rsp+A0h] [rbp-150h] BYREF
  __int64 *v33; // [rsp+B0h] [rbp-140h] BYREF
  __int64 v34; // [rsp+C0h] [rbp-130h] BYREF
  int v35; // [rsp+D0h] [rbp-120h]
  _QWORD v36[2]; // [rsp+F0h] [rbp-100h] BYREF
  __int64 v37; // [rsp+100h] [rbp-F0h] BYREF
  __int64 v38[2]; // [rsp+130h] [rbp-C0h] BYREF
  _QWORD v39[4]; // [rsp+140h] [rbp-B0h] BYREF
  char v40[8]; // [rsp+160h] [rbp-90h] BYREF
  __int64 *v41; // [rsp+168h] [rbp-88h]
  __int64 v42; // [rsp+178h] [rbp-78h] BYREF
  __int64 *v43; // [rsp+188h] [rbp-68h]
  __int64 v44; // [rsp+198h] [rbp-58h] BYREF
  _QWORD *v45; // [rsp+1A8h] [rbp-48h]
  _QWORD *v46; // [rsp+1B0h] [rbp-40h]
  __int64 v47; // [rsp+1B8h] [rbp-38h]

  LOWORD(v39[0]) = 260;
  v38[0] = a3 + 240;
  sub_16E1010(&v33);
  v4 = sub_1632FA0(a3);
  v5 = "nvptx64";
  if ( 8 * (unsigned int)sub_15A9520(v4, 0) != 64 )
    v5 = "nvptx";
  v6 = strlen(v5);
  v28[0] = v29;
  v28[1] = 0;
  LOBYTE(v29[0]) = 0;
  v38[0] = (__int64)v39;
  sub_12EA3F0(v38, v5, (__int64)&v5[v6]);
  v22 = sub_16D3AC0(v38, v28);
  if ( (_QWORD *)v38[0] != v39 )
    j_j___libc_free_0(v38[0], v39[0] + 1LL);
  if ( v22 )
  {
    v31 = 0;
    v30 = v32;
    LOBYTE(v32[0]) = 0;
    v38[0] = 0;
    v38[1] = 1;
    v39[0] = 8;
    v39[1] = 1;
    v39[2] = 1;
    v39[3] = 0;
    sub_167F890(v40);
    v7 = 0;
    if ( v35 )
    {
      v11 = *(_QWORD *)(a3 + 240);
      v12 = *(_QWORD *)(a2 + 1080);
      v26[1] = *(_QWORD *)(a3 + 248);
      v17 = v30;
      v13 = *(_QWORD *)(v12 + 16);
      v14 = *(_QWORD *)(v12 + 8);
      v26[0] = v11;
      v18 = v31;
      v19 = v14;
      v20 = v13;
      v15 = *(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64, _QWORD *, __int64, __int64 *, _BYTE *, __int64 *, __int64, _QWORD))(v22 + 88);
      if ( v15 )
      {
        v24[4] = 0;
        v25 = 0x100000000LL;
        v27 = 261;
        v26[2] = v26;
        sub_16E1010(v36);
        v16 = v15(v22, v36, v19, v20, v17, v18, v38, v24, &v25, 3, 0);
        v7 = v16;
        if ( (__int64 *)v36[0] != &v37 )
        {
          v23 = v16;
          j_j___libc_free_0(v36[0], v37 + 1);
          v7 = v23;
        }
      }
    }
    v8 = v46;
    v9 = v45;
    *a1 = v7;
    if ( v8 != v9 )
    {
      do
      {
        if ( (_QWORD *)*v9 != v9 + 2 )
          j_j___libc_free_0(*v9, v9[2] + 1LL);
        v9 += 4;
      }
      while ( v8 != v9 );
      v9 = v45;
    }
    if ( v9 )
      j_j___libc_free_0(v9, v47 - (_QWORD)v9);
    if ( v43 != &v44 )
      j_j___libc_free_0(v43, v44 + 1);
    if ( v41 != &v42 )
      j_j___libc_free_0(v41, v42 + 1);
    if ( v30 != v32 )
      j_j___libc_free_0(v30, v32[0] + 1LL);
  }
  else
  {
    v38[0] = (__int64)v39;
    sub_12EA3F0(v38, "Failed to locate nvptx target\n", (__int64)"");
    sub_1C3EFD0(v38, 1);
    if ( (_QWORD *)v38[0] != v39 )
      j_j___libc_free_0(v38[0], v39[0] + 1LL);
    *a1 = 0;
  }
  if ( (_QWORD *)v28[0] != v29 )
    j_j___libc_free_0(v28[0], v29[0] + 1LL);
  if ( v33 != &v34 )
    j_j___libc_free_0(v33, v34 + 1);
  return a1;
}

// Function: sub_24657C0
// Address: 0x24657c0
//
_BYTE *__fastcall sub_24657C0(__int64 *a1, __int64 a2, unsigned int **a3, __int64 a4, char a5)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r13
  __int64 v11; // rdx
  int v12; // eax
  unsigned __int64 *v13; // rdx
  unsigned __int64 *v14; // r13
  __int64 v15; // rdx
  unsigned __int64 v16; // r11
  unsigned __int64 v17; // r13
  _BYTE *v18; // rcx
  __int64 v19; // rax
  bool v20; // zf
  __int64 v21; // r13
  __int64 v22; // rsi
  __int64 v23; // rax
  _BYTE *v24; // rax
  _BYTE *v25; // rbx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  unsigned __int64 v31; // rsi
  __int64 v32; // r10
  _BYTE *v33; // rax
  unsigned __int64 v35; // [rsp+8h] [rbp-98h]
  _BYTE *v36; // [rsp+8h] [rbp-98h]
  unsigned __int64 v37; // [rsp+10h] [rbp-90h] BYREF
  char v38; // [rsp+18h] [rbp-88h]
  _BYTE *v39; // [rsp+20h] [rbp-80h] BYREF
  _BYTE *v40; // [rsp+28h] [rbp-78h]
  __int64 v41; // [rsp+30h] [rbp-70h]
  _QWORD v42[4]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v43; // [rsp+60h] [rbp-40h]

  v8 = sub_B2BEC0(*a1);
  v9 = sub_9208B0(v8, a4);
  v10 = a1[1];
  v42[0] = v9;
  v42[1] = v11;
  v37 = (unsigned __int64)(v9 + 7) >> 3;
  v38 = v11;
  v12 = sub_CA1930(&v37);
  v13 = (unsigned __int64 *)(v10 + 600);
  v14 = (unsigned __int64 *)(v10 + 536);
  if ( a5 )
    v14 = v13;
  v15 = *(_QWORD *)(a1[1] + 96);
  if ( v12 == 4 )
  {
    v16 = v14[4];
    v17 = v14[5];
  }
  else if ( v12 > 4 )
  {
    if ( v12 != 8 )
      goto LABEL_16;
    v16 = v14[6];
    v17 = v14[7];
  }
  else
  {
    if ( v12 != 1 )
    {
      if ( v12 == 2 )
      {
        v16 = v14[2];
        v17 = v14[3];
        goto LABEL_8;
      }
LABEL_16:
      v43 = 257;
      v18 = sub_94BCF0(a3, a2, v15, (__int64)v42);
      goto LABEL_17;
    }
    v16 = *v14;
    v17 = v14[1];
  }
LABEL_8:
  v35 = v16;
  v43 = 257;
  v18 = sub_94BCF0(a3, a2, v15, (__int64)v42);
  if ( !v17 )
  {
LABEL_17:
    v36 = v18;
    v28 = sub_CA1930(&v37);
    v29 = sub_AD64C0(*(_QWORD *)(a1[1] + 80), v28, 0);
    v30 = a1[1];
    if ( a5 )
    {
      v31 = *(_QWORD *)(v30 + 520);
      v32 = *(_QWORD *)(v30 + 528);
    }
    else
    {
      v31 = *(_QWORD *)(v30 + 504);
      v32 = *(_QWORD *)(v30 + 512);
    }
    v20 = *(_DWORD *)(v30 + 48) == 33;
    v43 = 257;
    if ( !v20 )
    {
      v39 = v36;
      v40 = (_BYTE *)v29;
      v21 = sub_921880(a3, v31, v32, (int)&v39, 2, (__int64)v42, 0);
      goto LABEL_11;
    }
    v33 = *(_BYTE **)(v30 + 680);
    v40 = v36;
    v41 = v29;
    v39 = v33;
    sub_921880(a3, v31, v32, (int)&v39, 3, (__int64)v42, 0);
    goto LABEL_15;
  }
  v19 = a1[1];
  v20 = *(_DWORD *)(v19 + 48) == 33;
  v43 = 257;
  if ( v20 )
  {
    v39 = *(_BYTE **)(v19 + 680);
    v40 = v18;
    sub_921880(a3, v35, v17, (int)&v39, 2, (__int64)v42, 0);
LABEL_15:
    v27 = a1[1];
    v43 = 257;
    v21 = sub_A82CA0(a3, *(_QWORD *)(v27 + 496), *(_QWORD *)(v27 + 680), 0, 0, (__int64)v42);
    goto LABEL_11;
  }
  v39 = v18;
  v21 = sub_921880(a3, v35, v17, (int)&v39, 1, (__int64)v42, 0);
LABEL_11:
  LODWORD(v39) = 0;
  v43 = 257;
  v22 = sub_94D3D0(a3, v21, (__int64)&v39, 1, (__int64)v42);
  v23 = a1[1];
  v43 = 257;
  v24 = sub_94BCF0(a3, v22, *(_QWORD *)(v23 + 96), (__int64)v42);
  v43 = 257;
  v25 = v24;
  LODWORD(v39) = 1;
  sub_94D3D0(a3, v21, (__int64)&v39, 1, (__int64)v42);
  return v25;
}

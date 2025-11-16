// Function: sub_2980CF0
// Address: 0x2980cf0
//
__int64 __fastcall sub_2980CF0(unsigned __int64 *a1, _BYTE *a2, _BYTE *a3, __int64 a4)
{
  __int64 v8; // rax
  __int64 v9; // rdi
  _BYTE *v10; // rsi
  __int64 v11; // r14
  __int64 *v12; // rax
  unsigned __int64 v14; // r9
  bool v15; // al
  _BYTE *v16; // r9
  _BYTE *v17; // rax
  unsigned __int64 v18; // [rsp+0h] [rbp-F0h]
  _BYTE *v19; // [rsp+0h] [rbp-F0h]
  _BYTE *v20; // [rsp+0h] [rbp-F0h]
  _BYTE *v21; // [rsp+0h] [rbp-F0h]
  bool v22; // [rsp+Fh] [rbp-E1h]
  unsigned __int64 v23; // [rsp+10h] [rbp-E0h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-D8h]
  unsigned int v25; // [rsp+20h] [rbp-D0h]
  unsigned __int64 v26; // [rsp+28h] [rbp-C8h]
  unsigned int v27; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v28; // [rsp+40h] [rbp-B0h] BYREF
  unsigned __int64 v29; // [rsp+48h] [rbp-A8h]
  unsigned int v30; // [rsp+50h] [rbp-A0h]
  unsigned __int64 v31; // [rsp+58h] [rbp-98h]
  unsigned int v32; // [rsp+60h] [rbp-90h]
  __m128i v33; // [rsp+70h] [rbp-80h] BYREF
  __int64 v34; // [rsp+80h] [rbp-70h]
  __int64 v35; // [rsp+88h] [rbp-68h]
  __int64 v36; // [rsp+90h] [rbp-60h]
  __int64 v37; // [rsp+98h] [rbp-58h]
  __int64 v38; // [rsp+A0h] [rbp-50h]
  __int64 v39; // [rsp+A8h] [rbp-48h]
  __int16 v40; // [rsp+B0h] [rbp-40h]

  if ( *a2 == 42 )
  {
    v10 = (_BYTE *)*((_QWORD *)a2 - 8);
    v11 = *((_QWORD *)a2 - 4);
    if ( v10 )
    {
      if ( *(_BYTE *)v11 != 17 )
      {
        if ( *v10 != 17 )
          goto LABEL_3;
        v17 = v10;
        v10 = (_BYTE *)*((_QWORD *)a2 - 4);
        v11 = (__int64)v17;
      }
      v9 = a1[2];
      goto LABEL_4;
    }
LABEL_26:
    if ( v11 )
      BUG();
    goto LABEL_3;
  }
  if ( *a2 == 58 )
  {
    v14 = *((_QWORD *)a2 - 8);
    v11 = *((_QWORD *)a2 - 4);
    if ( v14 )
    {
      if ( *(_BYTE *)v11 != 17 )
      {
        if ( *(_BYTE *)v14 != 17 )
          goto LABEL_3;
        v14 = *((_QWORD *)a2 - 4);
        v11 = *((_QWORD *)a2 - 8);
      }
      v18 = v14;
      v33 = (__m128i)*a1;
      v40 = 257;
      v34 = 0;
      v28 = v11 & 0xFFFFFFFFFFFFFFFBLL;
      v35 = 0;
      v36 = 0;
      v37 = 0;
      v38 = 0;
      v39 = 0;
      v30 = 1;
      v29 = 0;
      v32 = 1;
      v31 = 0;
      v23 = v14 & 0xFFFFFFFFFFFFFFFBLL;
      v25 = 1;
      v24 = 0;
      v27 = 1;
      v26 = 0;
      v15 = sub_9ACC00((__int64)&v23, (__int64)&v28, &v33);
      v16 = (_BYTE *)v18;
      v22 = v15;
      if ( v27 > 0x40 && v26 )
      {
        j_j___libc_free_0_0(v26);
        v16 = (_BYTE *)v18;
      }
      if ( v25 > 0x40 && v24 )
      {
        v19 = v16;
        j_j___libc_free_0_0(v24);
        v16 = v19;
      }
      if ( v32 > 0x40 && v31 )
      {
        v20 = v16;
        j_j___libc_free_0_0(v31);
        v16 = v20;
      }
      if ( v30 > 0x40 && v29 )
      {
        v21 = v16;
        j_j___libc_free_0_0(v29);
        v16 = v21;
      }
      if ( v22 )
      {
        v9 = a1[2];
        v10 = v16;
        goto LABEL_4;
      }
      goto LABEL_3;
    }
    goto LABEL_26;
  }
LABEL_3:
  v8 = sub_ACD640(*(_QWORD *)(a4 + 8), 0, 0);
  v9 = a1[2];
  v10 = a2;
  v11 = v8;
LABEL_4:
  v12 = sub_DD8400(v9, (__int64)v10);
  return sub_297F050((__int64)a1, 2, (__int64)v12, v11, a3, a4);
}

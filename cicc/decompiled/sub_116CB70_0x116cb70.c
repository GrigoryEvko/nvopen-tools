// Function: sub_116CB70
// Address: 0x116cb70
//
__int64 __fastcall sub_116CB70(char a1, unsigned __int8 a2, __int64 a3, _QWORD *a4)
{
  __int64 v4; // r13
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v11; // r12
  __int64 v12; // rax
  __int16 v13; // ax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 *v16; // r13
  __int64 v17; // r12
  unsigned int **v18; // rbx
  const char *v19; // rax
  __int64 v20; // rdx
  unsigned int *v21; // rax
  __int64 v22; // rbx
  unsigned int *v23; // r15
  __int64 v24; // rdx
  __int64 v25; // [rsp+0h] [rbp-280h]
  __int64 v26; // [rsp+8h] [rbp-278h]
  __int64 v27; // [rsp+8h] [rbp-278h]
  _QWORD v28[3]; // [rsp+10h] [rbp-270h] BYREF
  char v29; // [rsp+28h] [rbp-258h]
  __int64 v30[4]; // [rsp+30h] [rbp-250h] BYREF
  __int16 v31; // [rsp+50h] [rbp-230h]
  __int64 v32; // [rsp+60h] [rbp-220h] BYREF
  _QWORD v33[4]; // [rsp+68h] [rbp-218h] BYREF
  __int16 v34; // [rsp+88h] [rbp-1F8h]
  __int64 v35; // [rsp+90h] [rbp-1F0h] BYREF
  _QWORD v36[2]; // [rsp+A0h] [rbp-1E0h] BYREF
  char v37; // [rsp+B0h] [rbp-1D0h] BYREF
  char *v38; // [rsp+130h] [rbp-150h]
  char v39; // [rsp+140h] [rbp-140h] BYREF
  void *v40; // [rsp+1B0h] [rbp-D0h]
  __int64 v41[8]; // [rsp+1C0h] [rbp-C0h] BYREF
  char v42; // [rsp+200h] [rbp-80h]
  __int64 v43; // [rsp+208h] [rbp-78h]
  unsigned int v44; // [rsp+210h] [rbp-70h]

  v4 = 0;
  if ( !(_BYTE)qword_4F90988 )
    return v4;
  v25 = a4[10];
  v26 = a4[11];
  v8 = sub_BD5C60(a3);
  sub_1168EC0((__int64)v36, v8, v26, v25, a1);
  v9 = (__int64)v36;
  sub_116CAE0((__int64)v28, (__int64)v36, a3, a2);
  if ( v29 )
  {
    v33[0] = 0;
    v11 = a4[4];
    v33[1] = 0;
    v12 = *(_QWORD *)(v11 + 48);
    v32 = v11;
    v33[2] = v12;
    if ( v12 != 0 && v12 != -4096 && v12 != -8192 )
      sub_BD73F0((__int64)v33);
    v13 = *(_WORD *)(v11 + 64);
    v33[3] = *(_QWORD *)(v11 + 56);
    v34 = v13;
    sub_B33910(&v35, (__int64 *)v11);
    v14 = a4[4];
    *(_WORD *)(v14 + 64) = 0;
    *(_QWORD *)(v14 + 48) = 0;
    *(_QWORD *)(v14 + 56) = 0;
    v15 = a4[4];
    v30[0] = 0;
    sub_93FB40(v15, 0);
    v9 = v30[0];
    if ( v30[0] )
      sub_B91220((__int64)v30, v30[0]);
    v16 = (__int64 *)v28[0];
    v27 = v28[0] + 8LL * v28[1];
    if ( v28[0] != v27 )
    {
      do
      {
        v17 = *v16;
        v18 = (unsigned int **)a4[4];
        v19 = sub_BD5D20(*v16);
        v9 = v17;
        v30[1] = v20;
        v31 = 261;
        v30[0] = (__int64)v19;
        (*(void (__fastcall **)(unsigned int *, __int64, __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)v18[11] + 16LL))(
          v18[11],
          v17,
          v30,
          v18[7],
          v18[8]);
        v21 = *v18;
        v22 = (__int64)&(*v18)[4 * *((unsigned int *)v18 + 2)];
        if ( v21 != (unsigned int *)v22 )
        {
          v23 = v21;
          do
          {
            v24 = *((_QWORD *)v23 + 1);
            v9 = *v23;
            v23 += 4;
            sub_B99FD0(v17, v9, v24);
          }
          while ( (unsigned int *)v22 != v23 );
        }
        ++v16;
      }
      while ( (__int64 *)v27 != v16 );
    }
    v4 = v28[2];
    sub_F11320((__int64)&v32);
    if ( (v42 & 1) != 0 )
      goto LABEL_4;
    goto LABEL_9;
  }
  if ( (v42 & 1) == 0 )
  {
LABEL_9:
    v9 = 16LL * v44;
    sub_C7D6A0(v43, v9, 8);
  }
LABEL_4:
  sub_B32BF0(v41);
  v40 = &unk_49D94D0;
  nullsub_63();
  if ( v38 != &v39 )
    _libc_free(v38, v9);
  if ( (char *)v36[0] != &v37 )
    _libc_free(v36[0], v9);
  return v4;
}

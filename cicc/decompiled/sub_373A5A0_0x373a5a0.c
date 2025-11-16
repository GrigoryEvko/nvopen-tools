// Function: sub_373A5A0
// Address: 0x373a5a0
//
void __fastcall sub_373A5A0(__int64 *a1, __int64 a2, int **a3)
{
  int *v3; // r12
  __int64 v5; // rdi
  __int64 **v6; // r13
  __int16 v7; // ax
  _QWORD *v8; // rax
  char *v9; // rdi
  int v10; // r13d
  __int64 v11; // rax
  _QWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r9
  unsigned int v16; // r13d
  _BYTE *v17; // rdi
  size_t v18; // rdx
  __int64 v19; // [rsp+20h] [rbp-140h]
  __int16 v20; // [rsp+2Eh] [rbp-132h]
  __int64 *v21; // [rsp+30h] [rbp-130h]
  __int64 v22; // [rsp+48h] [rbp-118h]
  __int64 v23; // [rsp+50h] [rbp-110h] BYREF
  _BYTE *v24; // [rsp+58h] [rbp-108h] BYREF
  __int64 v25; // [rsp+60h] [rbp-100h]
  _BYTE v26[56]; // [rsp+68h] [rbp-F8h] BYREF
  _BYTE v27[4]; // [rsp+A0h] [rbp-C0h] BYREF
  int v28; // [rsp+A4h] [rbp-BCh]
  char *v29; // [rsp+B8h] [rbp-A8h]
  char v30; // [rsp+C8h] [rbp-98h] BYREF
  __int16 v31; // [rsp+104h] [rbp-5Ch]
  __int64 **v32; // [rsp+110h] [rbp-50h]

  v3 = *a3;
  v19 = (__int64)&(*a3)[22 * *((unsigned int *)a3 + 2)];
  if ( (int *)v19 != *a3 )
  {
    v21 = a1 + 11;
    do
    {
      v10 = *v3;
      v20 = sub_3736180((__int64)a1, 0x49u);
      v11 = sub_A777F0(0x30u, v21);
      v12 = (_QWORD *)v11;
      if ( v11 )
      {
        v22 = v11;
        *(_QWORD *)v11 = v11 | 4;
        *(_QWORD *)(v11 + 8) = 0;
        *(_QWORD *)(v11 + 16) = 0;
        *(_DWORD *)(v11 + 24) = -1;
        *(_WORD *)(v11 + 28) = v20;
        *(_BYTE *)(v11 + 30) = 0;
        *(_QWORD *)(v11 + 32) = 0;
        *(_QWORD *)(v11 + 40) = 0;
      }
      else
      {
        v22 = 0;
      }
      sub_324C1A0((__int64)a1, v11);
      v27[0] = 1;
      v28 = v10;
      sub_3738310(a1, (__int64)v12, 2, (__int64)v27);
      v13 = sub_A777F0(0x10u, v21);
      if ( v13 )
      {
        *(_QWORD *)v13 = 0;
        *(_DWORD *)(v13 + 8) = 0;
      }
      sub_3247620((__int64)v27, a1[23], (__int64)a1, v13);
      v31 |= 0x100u;
      v23 = *((_QWORD *)v3 + 1);
      v24 = v26;
      v25 = 0x200000000LL;
      v16 = v3[6];
      if ( !v16 || &v24 == (_BYTE **)(v3 + 4) )
        goto LABEL_3;
      if ( v16 > 2 )
      {
        sub_C8D5F0((__int64)&v24, v26, v16, 0x18u, v14, v15);
        v17 = v24;
        v18 = 24LL * (unsigned int)v3[6];
        if ( !v18 )
          goto LABEL_19;
      }
      else
      {
        v17 = v26;
        v18 = 24LL * v16;
      }
      memcpy(v17, *((const void **)v3 + 2), v18);
LABEL_19:
      LODWORD(v25) = v16;
LABEL_3:
      v5 = a1[23];
      v26[48] = *((_BYTE *)v3 + 80);
      sub_32200A0(v5, 0, (__int64)&v23, (unsigned __int64)v27);
      if ( v24 != v26 )
        _libc_free((unsigned __int64)v24);
      sub_3243D40((__int64)v27);
      v6 = v32;
      v7 = sub_37361D0((__int64)a1, 0x7Eu);
      sub_3249620(a1, (__int64)v12, v7, v6);
      v12[5] = a2 & 0xFFFFFFFFFFFFFFFBLL;
      v8 = *(_QWORD **)(a2 + 32);
      if ( v8 )
      {
        *v12 = *v8;
        **(_QWORD **)(a2 + 32) = v22 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v9 = v29;
      *(_QWORD *)(a2 + 32) = v22;
      if ( v9 != &v30 )
        _libc_free((unsigned __int64)v9);
      v3 += 22;
    }
    while ( (int *)v19 != v3 );
  }
}

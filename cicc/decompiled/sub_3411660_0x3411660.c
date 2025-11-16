// Function: sub_3411660
// Address: 0x3411660
//
unsigned __int8 *__fastcall sub_3411660(__int64 a1, unsigned int *a2, unsigned __int64 a3, __int64 a4, __int64 a5)
{
  unsigned int *v5; // rbx
  _QWORD *v7; // r9
  __int64 v9; // rdx
  __int64 v10; // rax
  unsigned int *v11; // r15
  __int64 v12; // rdx
  __int64 v13; // r11
  unsigned __int16 *v14; // rax
  unsigned int *v15; // rax
  __int64 v16; // rdx
  unsigned __int8 *v17; // r12
  __int128 v19; // [rsp-10h] [rbp-C0h]
  __int64 v20; // [rsp+8h] [rbp-A8h]
  _QWORD *v21; // [rsp+10h] [rbp-A0h]
  __int64 v22; // [rsp+10h] [rbp-A0h]
  __int64 v23; // [rsp+18h] [rbp-98h]
  _QWORD *v24; // [rsp+20h] [rbp-90h]
  unsigned __int16 *v26; // [rsp+30h] [rbp-80h] BYREF
  __int64 v27; // [rsp+38h] [rbp-78h]
  _BYTE v28[112]; // [rsp+40h] [rbp-70h] BYREF

  v5 = a2;
  if ( a3 == 1 )
    return *(unsigned __int8 **)a2;
  v7 = (_QWORD *)a1;
  v26 = (unsigned __int16 *)v28;
  v9 = 0;
  v27 = 0x400000000LL;
  v10 = 0;
  if ( a3 > 4 )
  {
    v22 = a5;
    sub_C8D5F0((__int64)&v26, v28, a3, 0x10u, a5, a1);
    v9 = (unsigned int)v27;
    a5 = v22;
    v7 = (_QWORD *)a1;
    v10 = (unsigned int)v27;
  }
  v11 = &a2[4 * a3];
  if ( a2 != v11 )
  {
    do
    {
      v12 = *(_QWORD *)(*(_QWORD *)v5 + 48LL) + 16LL * v5[2];
      v13 = *(_QWORD *)(v12 + 8);
      LOWORD(a5) = *(_WORD *)v12;
      if ( v10 + 1 > (unsigned __int64)HIDWORD(v27) )
      {
        v20 = a5;
        v21 = v7;
        v23 = *(_QWORD *)(v12 + 8);
        sub_C8D5F0((__int64)&v26, v28, v10 + 1, 0x10u, a5, (__int64)v7);
        v10 = (unsigned int)v27;
        a5 = v20;
        v7 = v21;
        v13 = v23;
      }
      v14 = &v26[8 * v10];
      v5 += 4;
      *(_QWORD *)v14 = a5;
      *((_QWORD *)v14 + 1) = v13;
      v10 = (unsigned int)(v27 + 1);
      LODWORD(v27) = v27 + 1;
    }
    while ( v11 != v5 );
    v9 = (unsigned int)v10;
  }
  v24 = v7;
  v15 = (unsigned int *)sub_33E5830(v7, v26, v9);
  *((_QWORD *)&v19 + 1) = a3;
  *(_QWORD *)&v19 = a2;
  v17 = sub_3411630(v24, 55, a4, v15, v16, (__int64)v24, v19);
  if ( v26 != (unsigned __int16 *)v28 )
    _libc_free((unsigned __int64)v26);
  return v17;
}

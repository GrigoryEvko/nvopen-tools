// Function: sub_D753A0
// Address: 0xd753a0
//
__int64 *__fastcall sub_D753A0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _BYTE *v7; // rdx
  __int64 *v8; // rax
  _QWORD *v9; // r13
  _QWORD *v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  unsigned int i; // [rsp+4h] [rbp-7Ch]
  unsigned __int64 v17[2]; // [rsp+10h] [rbp-70h] BYREF
  _BYTE *v18; // [rsp+20h] [rbp-60h]
  char v19[80]; // [rsp+30h] [rbp-50h] BYREF

  v6 = a2[2];
  for ( i = a4; v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v7 = *(_BYTE **)(v6 + 24);
    if ( *v7 == 28 )
    {
      v17[0] = 0;
      v17[1] = 0;
      v18 = v7;
      if ( v7 != (_BYTE *)-8192LL && v7 != (_BYTE *)-4096LL )
        sub_BD73F0((__int64)v17);
      sub_D6C8F0((__int64)v19, (__int64)(a1 + 63), v17, a4, a5, a6);
      LOBYTE(a4) = v18 + 4096 != 0;
      if ( ((unsigned __int8)a4 & (v18 != 0)) != 0 && v18 != (_BYTE *)-8192LL )
        sub_BD60C0(v17);
    }
  }
  v8 = a2 - 8;
  if ( *(_BYTE *)a2 == 26 )
    v8 = a2 - 4;
  sub_BD84D0((__int64)a2, *v8);
  sub_1041EA0(*a1, a2, a3, i);
  if ( *(_BYTE *)a2 == 27 )
    sub_D75120(a1, a2, 1);
  else
    sub_D73680(a1, (__int64)a2, 1);
  v9 = (_QWORD *)a1[63];
  v10 = &v9[3 * *((unsigned int *)a1 + 128)];
  while ( v9 != v10 )
  {
    while ( 1 )
    {
      v11 = *(v10 - 1);
      v10 -= 3;
      if ( v11 == 0 || v11 == -4096 || v11 == -8192 )
        break;
      sub_BD60C0(v10);
      if ( v9 == v10 )
        goto LABEL_19;
    }
  }
LABEL_19:
  v12 = a1[91];
  *((_DWORD *)a1 + 128) = 0;
  while ( v12 )
  {
    v13 = v12;
    sub_D690E0(*(_QWORD **)(v12 + 24));
    v12 = *(_QWORD *)(v12 + 16);
    sub_D68D70((_QWORD *)(v13 + 32));
    j_j___libc_free_0(v13, 56);
  }
  a1[91] = 0;
  a1[92] = (__int64)(a1 + 90);
  a1[93] = (__int64)(a1 + 90);
  a1[94] = 0;
  return a1 + 90;
}

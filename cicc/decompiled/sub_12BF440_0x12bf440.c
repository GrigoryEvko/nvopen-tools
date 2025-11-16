// Function: sub_12BF440
// Address: 0x12bf440
//
__int64 __fastcall sub_12BF440(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // rsi
  __int64 v4; // rax
  _BYTE *v5; // rbx
  __int64 v6; // rax
  _QWORD *v7; // r12
  __int64 v8; // rdx
  _QWORD *v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // rcx
  _QWORD *v13; // rbx
  _QWORD *v15; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v16; // [rsp+8h] [rbp-98h] BYREF
  __int64 v17; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-88h] BYREF
  _BYTE *v19; // [rsp+20h] [rbp-80h] BYREF
  __int64 v20; // [rsp+28h] [rbp-78h]
  _BYTE v21[112]; // [rsp+30h] [rbp-70h] BYREF

  v15 = &v19;
  v2 = *a2;
  *a2 = 0;
  v3 = &v17;
  v19 = v21;
  v17 = v2 | 1;
  v20 = 0x200000000LL;
  v16 = 0;
  sub_12BF260((__int64 *)&v18, &v17, (__int64 *)&v15);
  if ( (v18 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    v18 = v18 & 0xFFFFFFFFFFFFFFFELL | 1;
    sub_16BCAE0(&v18);
  }
  if ( (v17 & 1) != 0 || (v17 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v17);
  if ( (v16 & 1) != 0 || (v16 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v16);
  v4 = (unsigned int)v20;
  v5 = v19;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)a1 = a1 + 16;
  v6 = 32 * v4;
  *(_QWORD *)(a1 + 8) = 0;
  v7 = &v5[v6];
  if ( v5 != &v5[v6] )
  {
    v8 = *((_QWORD *)v5 + 1);
    v9 = v5 + 32;
    sub_2240E30(a1, v8 + (v6 >> 5) - 1 + v8 * ((unsigned __int64)(v6 - 32) >> 5));
    v3 = (__int64 *)*(v9 - 4);
    sub_2241490(a1, v3, *(v9 - 3), v10);
    while ( v7 != v9 )
    {
      if ( *(_QWORD *)(a1 + 8) == 0x3FFFFFFFFFFFFFFFLL )
        sub_4262D8((__int64)"basic_string::append");
      v9 += 4;
      sub_2241490(a1, "\n", 1, v11);
      v3 = (__int64 *)*(v9 - 4);
      sub_2241490(a1, v3, *(v9 - 3), v12);
    }
    v13 = v19;
    v7 = &v19[32 * (unsigned int)v20];
    if ( v19 != (_BYTE *)v7 )
    {
      do
      {
        v7 -= 4;
        if ( (_QWORD *)*v7 != v7 + 2 )
        {
          v3 = (__int64 *)(v7[2] + 1LL);
          j_j___libc_free_0(*v7, v3);
        }
      }
      while ( v7 != v13 );
      v7 = v19;
    }
  }
  if ( v7 != (_QWORD *)v21 )
    _libc_free(v7, v3);
  return a1;
}

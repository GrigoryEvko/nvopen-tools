// Function: sub_22B6090
// Address: 0x22b6090
//
void __fastcall sub_22B6090(__int64 a1, __int64 a2, char **a3, char **a4, __int64 a5, unsigned __int64 a6)
{
  __int64 v6; // r12
  __int64 v8; // rsi
  int v9; // eax
  __int64 v10; // r8
  unsigned __int64 *v11; // r8
  unsigned __int64 *i; // rdi
  unsigned __int64 v13; // rax
  unsigned __int64 *v14; // rcx
  unsigned __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v19; // [rsp+20h] [rbp-80h] BYREF
  __int16 v20; // [rsp+28h] [rbp-78h]
  unsigned __int64 v21[4]; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 *v22; // [rsp+50h] [rbp-50h] BYREF
  unsigned __int64 *v23; // [rsp+58h] [rbp-48h]
  __int64 v24; // [rsp+60h] [rbp-40h]

  v6 = a2 + 48;
  v8 = *(_QWORD *)(a2 + 56);
  v19 = v8;
  v20 = 1;
  memset(v21, 0, 24);
  v22 = 0;
  v23 = 0;
  v24 = 0;
  if ( v8 == v6 )
  {
    if ( !*(_BYTE *)(a1 + 72) )
      goto LABEL_15;
  }
  else
  {
    do
    {
      if ( v8 )
        v8 -= 24;
      v9 = sub_22B3050(a1 + 104, v8);
      switch ( v9 )
      {
        case 1:
          sub_22B4880((unsigned int *)a1, &v19, (__int64)v21, (__int64)&v22, 0, a6);
          break;
        case 2:
          *(_BYTE *)(a1 + 72) = 0;
          break;
        case 0:
          sub_22B5D50(a1, &v19, (__int64)v21, (__int64)&v22, v10, a6);
          break;
      }
      v8 = *(_QWORD *)(v19 + 8);
      v20 = 0;
      v19 = v8;
    }
    while ( v6 != v8 );
    if ( !*(_BYTE *)(a1 + 72) )
      goto LABEL_13;
  }
  sub_22B4880((unsigned int *)a1, &v19, (__int64)v21, (__int64)&v22, 1, a6);
LABEL_13:
  v11 = v23;
  for ( i = v22; i != v11; *v14 = *v14 & 7 | v13 )
  {
    v13 = *i;
    v14 = *(unsigned __int64 **)(a1 + 96);
    ++i;
    v15 = *v14;
    v16 = *(_QWORD *)v13;
    *(_QWORD *)(v13 + 8) = v14;
    v15 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v13 = v15 | v16 & 7;
    *(_QWORD *)(v15 + 8) = v13;
  }
LABEL_15:
  sub_22B0250(a3, (__int64)&v22);
  sub_22B0470(a4, (__int64)v21);
  if ( v22 )
    j_j___libc_free_0((unsigned __int64)v22);
  if ( v21[0] )
    j_j___libc_free_0(v21[0]);
}

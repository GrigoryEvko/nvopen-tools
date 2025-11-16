// Function: sub_2D03BE0
// Address: 0x2d03be0
//
__int64 __fastcall sub_2D03BE0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v3; // r15
  int v4; // edx
  unsigned __int64 v5; // r9
  __int64 v7; // r12
  __int64 v8; // rbx
  _QWORD *v9; // rdi
  unsigned __int64 v10; // r9
  __int64 v11; // rcx
  __int64 v12; // r15
  __int64 v13; // rbx
  __int64 i; // rax
  size_t v15; // r13
  __int64 v16; // rax
  __int64 v17; // r8
  __int64 v18; // r14
  __int64 *v19; // r12
  __int64 v21; // [rsp+18h] [rbp-78h]
  __int64 v22; // [rsp+20h] [rbp-70h]
  unsigned __int64 v23; // [rsp+28h] [rbp-68h]
  __int64 v24; // [rsp+38h] [rbp-58h]
  unsigned __int64 v25; // [rsp+40h] [rbp-50h] BYREF
  __int128 v26; // [rsp+48h] [rbp-48h]

  v3 = a1;
  v4 = *((_DWORD *)a2 + 3);
  v25 = 0;
  *(_QWORD *)&v26 = 0;
  *((_QWORD *)&v26 + 1) = 0x1000000000LL;
  if ( !v4 )
  {
    v5 = 0;
    if ( (_BYTE)qword_5014E08 )
      goto LABEL_4;
    goto LABEL_3;
  }
  sub_C92620((__int64)&v25, *((_DWORD *)a2 + 2));
  v10 = v25;
  v11 = *a2;
  v23 = v25;
  v22 = *a2;
  *(_QWORD *)((char *)&v26 + 4) = *(__int64 *)((char *)a2 + 12);
  if ( !(_DWORD)v26 )
    goto LABEL_23;
  v12 = 0;
  v13 = 8LL * (unsigned int)v26 + 8;
  v24 = 8LL * (unsigned int)(v26 - 1);
  for ( i = v11; ; i = *a2 )
  {
    v18 = *(_QWORD *)(i + v12);
    v19 = (__int64 *)(v10 + v12);
    if ( v18 )
    {
      if ( v18 != -8 )
        break;
    }
    *v19 = v18;
    v13 += 4;
    if ( v12 == v24 )
      goto LABEL_22;
LABEL_19:
    v10 = v25;
    v12 += 8;
  }
  v15 = *(_QWORD *)v18;
  v16 = sub_C7D670(*(_QWORD *)v18 + 17LL, 8);
  v17 = v16;
  if ( v15 )
  {
    v21 = v16;
    memcpy((void *)(v16 + 16), (const void *)(v18 + 16), v15);
    v17 = v21;
  }
  *(_BYTE *)(v17 + v15 + 16) = 0;
  *(_QWORD *)v17 = v15;
  *(_DWORD *)(v17 + 8) = *(_DWORD *)(v18 + 8);
  *v19 = v17;
  *(_DWORD *)(v23 + v13) = *(_DWORD *)(v22 + v13);
  v13 += 4;
  if ( v12 != v24 )
    goto LABEL_19;
LABEL_22:
  v3 = a1;
LABEL_23:
  if ( (_BYTE)qword_5014E08 )
LABEL_4:
    sub_2D02FE0((__int64)&v25, a3);
  v5 = v25;
  if ( DWORD1(v26) && (_DWORD)v26 )
  {
    v7 = 8LL * (unsigned int)v26;
    v8 = 0;
    do
    {
      v9 = *(_QWORD **)(v5 + v8);
      if ( v9 != (_QWORD *)-8LL && v9 )
      {
        sub_C7D6A0((__int64)v9, *v9 + 17LL, 8);
        v5 = v25;
      }
      v8 += 8;
    }
    while ( v8 != v7 );
  }
LABEL_3:
  _libc_free(v5);
  *(_QWORD *)(v3 + 48) = 0;
  *(_QWORD *)(v3 + 8) = v3 + 32;
  *(_QWORD *)(v3 + 56) = v3 + 80;
  *(_QWORD *)(v3 + 16) = 0x100000002LL;
  *(_QWORD *)(v3 + 64) = 2;
  *(_QWORD *)(v3 + 32) = &qword_4F82400;
  *(_DWORD *)(v3 + 72) = 0;
  *(_BYTE *)(v3 + 76) = 1;
  *(_DWORD *)(v3 + 24) = 0;
  *(_BYTE *)(v3 + 28) = 1;
  *(_QWORD *)v3 = 1;
  return v3;
}

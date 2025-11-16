// Function: sub_33F99B0
// Address: 0x33f99b0
//
void __fastcall sub_33F99B0(__int64 a1, __int64 *a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  _BYTE *v15; // rdi
  _BYTE *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-88h]
  __int64 v18; // [rsp+8h] [rbp-88h]
  __int64 v19; // [rsp+10h] [rbp-80h]
  __int64 v20; // [rsp+10h] [rbp-80h]
  _BYTE *v21; // [rsp+20h] [rbp-70h] BYREF
  __int64 v22; // [rsp+28h] [rbp-68h]
  _BYTE v23[96]; // [rsp+30h] [rbp-60h] BYREF

  v22 = 0x600000000LL;
  v8 = *a2;
  v9 = a2[1];
  v21 = v23;
  v10 = v9 + 24 * v8;
  if ( v9 == v10 )
  {
    v12 = (__int64 *)a2[3];
    v13 = (__int64)&v12[a2[2]];
    if ( (__int64 *)v13 == v12 )
      goto LABEL_21;
    v14 = 6;
    v11 = 0;
  }
  else
  {
    v11 = 0;
    do
    {
      while ( *(_DWORD *)v9 )
      {
        v9 += 24;
        if ( v10 == v9 )
          goto LABEL_8;
      }
      a6 = *(_QWORD *)(v9 + 8);
      if ( v11 + 1 > (unsigned __int64)HIDWORD(v22) )
      {
        v18 = *(_QWORD *)(v9 + 8);
        v20 = v10;
        sub_C8D5F0((__int64)&v21, v23, v11 + 1, 8u, v10, a6);
        v11 = (unsigned int)v22;
        a6 = v18;
        v10 = v20;
      }
      v9 += 24;
      *(_QWORD *)&v21[8 * v11] = a6;
      v11 = (unsigned int)(v22 + 1);
      LODWORD(v22) = v22 + 1;
    }
    while ( v10 != v9 );
LABEL_8:
    v12 = (__int64 *)a2[3];
    v13 = (__int64)&v12[a2[2]];
    if ( v12 == (__int64 *)v13 )
      goto LABEL_14;
    v14 = HIDWORD(v22);
  }
  while ( 1 )
  {
    a6 = *v12;
    if ( v11 + 1 > v14 )
    {
      v19 = v13;
      v17 = *v12;
      sub_C8D5F0((__int64)&v21, v23, v11 + 1, 8u, v13, a6);
      v11 = (unsigned int)v22;
      a6 = v17;
      v13 = v19;
    }
    ++v12;
    *(_QWORD *)&v21[8 * v11] = a6;
    v11 = (unsigned int)(v22 + 1);
    LODWORD(v22) = v22 + 1;
    if ( (__int64 *)v13 == v12 )
      break;
    v14 = HIDWORD(v22);
  }
LABEL_14:
  v15 = v21;
  a4 = (__int64)&v21[8 * v11];
  if ( (_BYTE *)a4 != v21 )
  {
    v16 = v21;
    do
    {
      if ( *(_QWORD *)v16 )
        *(_BYTE *)(*(_QWORD *)v16 + 32LL) |= 1u;
      v16 += 8;
    }
    while ( (_BYTE *)a4 != v16 );
  }
  if ( v15 != v23 )
    _libc_free((unsigned __int64)v15);
LABEL_21:
  sub_33F9440(*(_QWORD *)(a1 + 720), a2, a3, a4, v13, a6);
}

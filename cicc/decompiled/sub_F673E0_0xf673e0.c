// Function: sub_F673E0
// Address: 0xf673e0
//
char __fastcall sub_F673E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r12
  _QWORD *v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // rbx
  _QWORD *v12; // rax
  _BYTE *v13; // rdx
  _QWORD *v14; // rbx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  _QWORD *v18; // rcx
  __int64 v19; // rdx
  __int64 v21; // [rsp+8h] [rbp-88h]
  _QWORD *v22; // [rsp+10h] [rbp-80h] BYREF
  __int64 v23; // [rsp+18h] [rbp-78h]
  _QWORD v24[14]; // [rsp+20h] [rbp-70h] BYREF

  v6 = a3;
  v7 = a2;
  LODWORD(a2) = 1;
  v22 = v24;
  v23 = 0x800000001LL;
  v24[0] = a1;
  v8 = v24;
  do
  {
    v9 = (unsigned int)a2;
    a2 = (unsigned int)(a2 - 1);
    v10 = *(_BYTE *)(v6 + 28) == 0;
    v11 = v8[v9 - 1];
    LODWORD(v23) = a2;
    if ( v10 )
      goto LABEL_12;
    v12 = *(_QWORD **)(v6 + 8);
    a4 = *(unsigned int *)(v6 + 20);
    a3 = (__int64)&v12[a4];
    if ( v12 != (_QWORD *)a3 )
    {
      while ( v11 != *v12 )
      {
        if ( (_QWORD *)a3 == ++v12 )
          goto LABEL_31;
      }
LABEL_7:
      v8 = v22;
      continue;
    }
LABEL_31:
    if ( (unsigned int)a4 >= *(_DWORD *)(v6 + 16) )
    {
LABEL_12:
      sub_C8CC70(v6, v11, a3, a4, a5, a6);
      a2 = (unsigned int)v23;
      LOBYTE(v12) = a3 & (v7 != v11);
    }
    else
    {
      LOBYTE(v12) = v7 != v11;
      a4 = (unsigned int)(a4 + 1);
      *(_DWORD *)(v6 + 20) = a4;
      *(_QWORD *)a3 = v11;
      a2 = (unsigned int)v23;
      ++*(_QWORD *)v6;
    }
    if ( !(_BYTE)v12 )
      goto LABEL_7;
    v12 = *(_QWORD **)(v11 + 16);
    do
    {
      if ( !v12 )
      {
        a3 = (unsigned int)a2;
        if ( HIDWORD(v23) < (unsigned int)a2 )
        {
          LOBYTE(v12) = sub_C8D5F0((__int64)&v22, v24, (unsigned int)a2, 8u, a5, a6);
          a3 = (unsigned int)v23;
        }
        v8 = v22;
        a5 = 0;
        goto LABEL_30;
      }
      v13 = (_BYTE *)v12[3];
      v14 = v12;
      v12 = (_QWORD *)v12[1];
    }
    while ( (unsigned __int8)(*v13 - 30) > 0xAu );
    v15 = v14;
    v16 = 0;
    while ( 1 )
    {
      v15 = (_QWORD *)v15[1];
      if ( !v15 )
        break;
      while ( (unsigned __int8)(*(_BYTE *)v15[3] - 30) <= 0xAu )
      {
        v15 = (_QWORD *)v15[1];
        ++v16;
        if ( !v15 )
          goto LABEL_21;
      }
    }
LABEL_21:
    a5 = v16 + 1;
    v17 = v16 + 1 + a2;
    if ( HIDWORD(v23) < v17 )
    {
      v21 = v16 + 1;
      sub_C8D5F0((__int64)&v22, v24, v17, 8u, a5, a6);
      LOBYTE(v12) = (_BYTE)v22;
      a5 = v21;
      v18 = &v22[(unsigned int)v23];
    }
    else
    {
      LOBYTE(v12) = (_BYTE)v22;
      v18 = &v22[a2];
    }
    v19 = v14[3];
LABEL_26:
    if ( v18 )
    {
      v12 = *(_QWORD **)(v19 + 40);
      *v18 = v12;
    }
    while ( 1 )
    {
      v14 = (_QWORD *)v14[1];
      if ( !v14 )
        break;
      v19 = v14[3];
      LOBYTE(v12) = *(_BYTE *)v19 - 30;
      if ( (unsigned __int8)v12 <= 0xAu )
      {
        ++v18;
        goto LABEL_26;
      }
    }
    a3 = (unsigned int)v23;
    v8 = v22;
LABEL_30:
    a4 = a3 + a5;
    LODWORD(v23) = a3 + a5;
    a2 = (unsigned int)(a3 + a5);
  }
  while ( (_DWORD)a2 );
  if ( v8 != v24 )
    LOBYTE(v12) = _libc_free(v8, a2);
  return (char)v12;
}

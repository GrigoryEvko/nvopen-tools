// Function: sub_274ADE0
// Address: 0x274ade0
//
__int64 __fastcall sub_274ADE0(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  int v4; // r8d
  __int64 v5; // r8
  _QWORD *v6; // r14
  __int64 v7; // r15
  unsigned __int8 *v8; // r9
  unsigned __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int8 **v13; // rax
  unsigned __int8 **v14; // rax
  unsigned __int8 *v15; // r12
  unsigned int v16; // r12d
  unsigned __int8 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // r14
  const void *v22; // r12
  unsigned int v23; // eax
  unsigned int v25; // [rsp+14h] [rbp-9Ch]
  _BYTE *v26; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v27; // [rsp+20h] [rbp-90h]
  _BYTE *v28; // [rsp+30h] [rbp-80h] BYREF
  __int64 v29; // [rsp+38h] [rbp-78h]
  _BYTE v30[112]; // [rsp+40h] [rbp-70h] BYREF

  v4 = *(_DWORD *)(a1 + 4);
  v28 = v30;
  v29 = 0x400000000LL;
  v5 = v4 & 0x7FFFFFF;
  if ( !(_DWORD)v5 )
    return 0;
  v6 = (_QWORD *)a1;
  v7 = 0;
  v8 = 0;
  v9 = v3;
  do
  {
    while ( 1 )
    {
      v14 = (unsigned __int8 **)(*(_QWORD *)(a1 - 8) + 32 * v7);
      v15 = *v14;
      if ( **v14 <= 0x15u )
      {
        v10 = (unsigned int)v29;
        v11 = v9 & 0xFFFFFFFF00000000LL | (unsigned int)v7;
        v12 = (unsigned int)v29 + 1LL;
        v9 = v11;
        if ( v12 > HIDWORD(v29) )
        {
          v25 = v5;
          v27 = v8;
          sub_C8D5F0((__int64)&v28, v30, v12, 0x10u, v5, (__int64)v8);
          v10 = (unsigned int)v29;
          v5 = v25;
          v8 = v27;
        }
        v13 = (unsigned __int8 **)&v28[16 * v10];
        *v13 = v15;
        v13[1] = (unsigned __int8 *)v11;
        LODWORD(v29) = v29 + 1;
        goto LABEL_6;
      }
      if ( !v8 )
        break;
      if ( v15 != v8 )
        goto LABEL_10;
LABEL_6:
      if ( (_DWORD)v5 == (_DWORD)++v7 )
        goto LABEL_15;
    }
    ++v7;
    v8 = *v14;
  }
  while ( (_DWORD)v5 != (_DWORD)v7 );
LABEL_15:
  v18 = v8;
  if ( !v8 )
    goto LABEL_10;
  v19 = (unsigned int)v29;
  if ( !(_DWORD)v29 )
    goto LABEL_10;
  v20 = *(_QWORD *)(a1 + 40);
  if ( *v8 <= 0x1Cu )
    goto LABEL_20;
  if ( !(unsigned __int8)sub_B19D00(a3, (__int64)v8, *(_QWORD *)(a1 + 40)) )
    goto LABEL_10;
  v19 = (unsigned int)v29;
LABEL_20:
  v26 = &v28[16 * v19];
  if ( v26 != v28 )
  {
    v21 = (unsigned __int64)v28;
    while ( 1 )
    {
      v22 = *(const void **)v21;
      if ( v22 != sub_22CF3A0(
                    a2,
                    (__int64)v18,
                    *(_QWORD *)(*(_QWORD *)(a1 - 8)
                              + 32LL * *(unsigned int *)(a1 + 72)
                              + 8LL * *(unsigned int *)(v21 + 8)),
                    v20,
                    a1) )
        goto LABEL_10;
      v21 += 16LL;
      if ( v26 == (_BYTE *)v21 )
      {
        v6 = (_QWORD *)a1;
        break;
      }
    }
  }
  LOBYTE(v23) = sub_98ED70(v18, 0, (__int64)v6, a3, 0);
  v16 = v23;
  if ( !(_BYTE)v23 )
  {
LABEL_10:
    v16 = 0;
    goto LABEL_11;
  }
  sub_BD84D0((__int64)v6, (__int64)v18);
  sub_B43D60(v6);
LABEL_11:
  if ( v28 != v30 )
    _libc_free((unsigned __int64)v28);
  return v16;
}

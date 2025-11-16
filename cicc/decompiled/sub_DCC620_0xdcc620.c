// Function: sub_DCC620
// Address: 0xdcc620
//
_QWORD *__fastcall sub_DCC620(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // r8
  __int64 v9; // rdx
  unsigned __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // r12
  _QWORD *v17; // [rsp+10h] [rbp-90h]
  _QWORD *v18; // [rsp+10h] [rbp-90h]
  __int64 v19; // [rsp+18h] [rbp-88h]
  _QWORD v20[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-70h] BYREF
  _BYTE *v22; // [rsp+40h] [rbp-60h] BYREF
  __int64 v23; // [rsp+48h] [rbp-58h]
  _BYTE v24[80]; // [rsp+50h] [rbp-50h] BYREF

  v22 = v24;
  v23 = 0x300000000LL;
  v2 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v2 == 1 )
  {
    v12 = **(_QWORD **)(a1 + 32);
    v13 = v24;
  }
  else
  {
    v3 = 8;
    v19 = 8LL * (unsigned int)(v2 - 2) + 16;
    do
    {
      v4 = *(_QWORD *)(a1 + 32);
      v5 = *(_QWORD *)(v4 + v3);
      v6 = *(_QWORD *)(v4 + v3 - 8);
      v20[0] = v21;
      v21[0] = v6;
      v21[1] = v5;
      v20[1] = 0x200000002LL;
      v7 = sub_DC7EB0(a2, (__int64)v20, 0, 0);
      if ( (_QWORD *)v20[0] != v21 )
      {
        v17 = v7;
        _libc_free(v20[0], v20);
        v7 = v17;
      }
      v9 = (unsigned int)v23;
      v10 = (unsigned int)v23 + 1LL;
      if ( v10 > HIDWORD(v23) )
      {
        v18 = v7;
        sub_C8D5F0((__int64)&v22, v24, (unsigned int)v23 + 1LL, 8u, v8, v10);
        v9 = (unsigned int)v23;
        v7 = v18;
      }
      v3 += 8;
      *(_QWORD *)&v22[8 * v9] = v7;
      v11 = (unsigned int)(v23 + 1);
      LODWORD(v23) = v23 + 1;
    }
    while ( v19 != v3 );
    v12 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a1 + 40) - 1));
    if ( v11 + 1 > (unsigned __int64)HIDWORD(v23) )
    {
      sub_C8D5F0((__int64)&v22, v24, v11 + 1, 8u, v8, v10);
      v13 = &v22[8 * (unsigned int)v23];
    }
    else
    {
      v13 = &v22[8 * v11];
    }
  }
  *v13 = v12;
  v14 = *(_QWORD *)(a1 + 48);
  LODWORD(v23) = v23 + 1;
  v15 = sub_DBFF60((__int64)a2, (unsigned int *)&v22, v14, 0);
  if ( v22 != v24 )
    _libc_free(v22, &v22);
  return v15;
}

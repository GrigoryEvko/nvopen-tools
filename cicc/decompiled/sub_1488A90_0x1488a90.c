// Function: sub_1488A90
// Address: 0x1488a90
//
__int64 __fastcall sub_1488A90(__int64 a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rax
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 *v18; // [rsp+10h] [rbp-90h]
  __int64 v19; // [rsp+10h] [rbp-90h]
  __int64 v20; // [rsp+18h] [rbp-88h]
  __int64 v21[2]; // [rsp+20h] [rbp-80h] BYREF
  _QWORD v22[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v23; // [rsp+40h] [rbp-60h] BYREF
  __int64 v24; // [rsp+48h] [rbp-58h]
  _BYTE v25[80]; // [rsp+50h] [rbp-50h] BYREF

  v23 = (__int64 *)v25;
  v24 = 0x300000000LL;
  v4 = *(_QWORD *)(a1 + 40);
  if ( (_DWORD)v4 == 1 )
  {
    v13 = **(_QWORD **)(a1 + 32);
    v14 = (__int64 *)v25;
  }
  else
  {
    v5 = 8;
    v20 = 8LL * (unsigned int)(v4 - 2) + 16;
    do
    {
      v6 = *(_QWORD *)(a1 + 32);
      v7 = *(_QWORD *)(v6 + v5);
      v8 = *(_QWORD *)(v6 + v5 - 8);
      v21[0] = (__int64)v22;
      v22[0] = v8;
      v22[1] = v7;
      v21[1] = 0x200000002LL;
      v9 = sub_147DD40(a2, v21, 0, 0, a3, a4);
      v10 = (__int64)v9;
      if ( (_QWORD *)v21[0] != v22 )
      {
        v18 = v9;
        _libc_free(v21[0]);
        v10 = (__int64)v18;
      }
      v11 = (unsigned int)v24;
      if ( (unsigned int)v24 >= HIDWORD(v24) )
      {
        v19 = v10;
        sub_16CD150(&v23, v25, 0, 8);
        v11 = (unsigned int)v24;
        v10 = v19;
      }
      v5 += 8;
      v23[v11] = v10;
      v12 = (unsigned int)(v24 + 1);
      LODWORD(v24) = v24 + 1;
    }
    while ( v20 != v5 );
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a1 + 40) - 1));
    if ( (unsigned int)v12 >= HIDWORD(v24) )
    {
      sub_16CD150(&v23, v25, 0, 8);
      v14 = &v23[(unsigned int)v24];
    }
    else
    {
      v14 = &v23[v12];
    }
  }
  *v14 = v13;
  v15 = *(_QWORD *)(a1 + 48);
  LODWORD(v24) = v24 + 1;
  v16 = sub_14785F0(a2, &v23, v15, 0);
  if ( v23 != (__int64 *)v25 )
    _libc_free((unsigned __int64)v23);
  return v16;
}

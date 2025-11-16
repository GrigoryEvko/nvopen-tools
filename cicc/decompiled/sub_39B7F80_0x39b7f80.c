// Function: sub_39B7F80
// Address: 0x39b7f80
//
__int64 __fastcall sub_39B7F80(
        __int64 *a1,
        unsigned int a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        int a6,
        unsigned int a7)
{
  int v7; // r15d
  __int64 *v8; // r12
  _QWORD *v9; // rbx
  unsigned int v10; // r13d
  __int64 result; // rax
  __int64 ***v12; // r14
  __int64 ***v13; // r13
  __int64 *v14; // r9
  __int64 v15; // rax
  int v16; // r14d
  unsigned int v17; // r14d
  char v18; // al
  __int64 v19; // rdx
  __int64 v20; // rcx
  _QWORD *v21; // r15
  int v22; // ebx
  int v23; // r12d
  __int64 v24; // rdx
  __int64 *v25; // [rsp+8h] [rbp-98h]
  unsigned int v28; // [rsp+30h] [rbp-70h]
  unsigned int v31; // [rsp+38h] [rbp-68h]
  __int64 *v32; // [rsp+40h] [rbp-60h] BYREF
  __int64 v33; // [rsp+48h] [rbp-58h]
  _BYTE v34[80]; // [rsp+50h] [rbp-50h] BYREF

  v7 = a2;
  v8 = a1;
  v9 = (_QWORD *)a3;
  v10 = a7;
  v28 = 1;
  if ( *(_BYTE *)(a3 + 8) == 16 )
    v28 = *(_DWORD *)(a3 + 32);
  if ( a2 == 128 )
    return 1;
  if ( a2 > 0x80 )
  {
    result = 1;
    if ( a2 == 130 )
      return result;
  }
  else if ( a2 - 83 <= 0xC )
  {
    v32 = *(__int64 **)*a4;
    return sub_39B6390(a1, a2, a3, (__int64 *)&v32, 1, a6, 0xFFFFFFFF);
  }
  v32 = (__int64 *)v34;
  v33 = 0x400000000LL;
  if ( &a4[a5] != a4 )
  {
    v12 = (__int64 ***)&a4[a5];
    v13 = (__int64 ***)a4;
    do
    {
      v14 = **v13;
      if ( a7 != 1 )
        v14 = sub_16463B0(**v13, a7);
      v15 = (unsigned int)v33;
      if ( (unsigned int)v33 >= HIDWORD(v33) )
      {
        v25 = v14;
        sub_16CD150((__int64)&v32, v34, 0, 8, a5, (int)v14);
        v15 = (unsigned int)v33;
        v14 = v25;
      }
      ++v13;
      v32[v15] = (__int64)v14;
      LODWORD(v33) = v33 + 1;
    }
    while ( v12 != v13 );
    v10 = a7;
    v9 = (_QWORD *)a3;
  }
  if ( v10 > 1 )
  {
    v16 = 0;
    if ( !*((_BYTE *)v9 + 8) )
    {
LABEL_19:
      v17 = sub_39B48D0(v8, a4, a5, v10) + v16;
      goto LABEL_20;
    }
    v9 = sub_16463B0(v9, v10);
LABEL_26:
    v18 = *((_BYTE *)v9 + 8);
    if ( v18 )
    {
      v19 = v9[4];
      v16 = 0;
      if ( (int)v19 > 0 )
      {
        v20 = 0;
        v21 = v9;
        v22 = 0;
        v23 = v19;
        while ( 1 )
        {
          v24 = (__int64)v21;
          if ( v18 == 16 )
            v24 = *(_QWORD *)v21[2];
          ++v22;
          v16 += sub_1F43D80(a1[2], *a1, v24, v20);
          if ( v23 == v22 )
            break;
          v18 = *((_BYTE *)v21 + 8);
        }
        v8 = a1;
        v9 = v21;
        v7 = a2;
      }
    }
    else
    {
      v16 = 0;
    }
    goto LABEL_19;
  }
  if ( v28 > 1 )
    goto LABEL_26;
  v17 = -1;
LABEL_20:
  result = sub_39B6390(v8, v7, (__int64)v9, v32, v33, a6, v17);
  if ( v32 != (__int64 *)v34 )
  {
    v31 = result;
    _libc_free((unsigned __int64)v32);
    return v31;
  }
  return result;
}

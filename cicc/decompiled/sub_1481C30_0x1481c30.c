// Function: sub_1481C30
// Address: 0x1481c30
//
__int64 __fastcall sub_1481C30(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rax
  __int64 *v5; // rbx
  __int64 *v6; // r14
  __int64 v8; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // r8
  __int64 v12; // rbx
  __int64 *v13; // r14
  __int64 v14; // r9
  __int64 v15; // rax
  __int64 result; // rax
  __int64 v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  _BYTE *v19; // [rsp+10h] [rbp-50h] BYREF
  __int64 v20; // [rsp+18h] [rbp-48h]
  _BYTE v21[64]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a2 + 8);
  v5 = *(__int64 **)a2;
  if ( v4 == 1 )
    return *v5;
  v6 = &v5[v4];
  if ( v6 == v5 )
  {
    v19 = v21;
    v20 = 0x200000000LL;
  }
  else
  {
    v8 = 0;
    do
    {
      while ( 1 )
      {
        v10 = *v5;
        if ( !v8 )
          break;
        v9 = sub_1456040(v10);
        ++v5;
        v8 = sub_1456E50((__int64)a1, v8, v9);
        if ( v6 == v5 )
          goto LABEL_7;
      }
      ++v5;
      v8 = sub_1456040(v10);
    }
    while ( v6 != v5 );
LABEL_7:
    v11 = *(__int64 **)a2;
    v12 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    v19 = v21;
    v20 = 0x200000000LL;
    if ( v11 != (__int64 *)v12 )
    {
      v13 = v11;
      do
      {
        v14 = sub_14758B0((__int64)a1, *v13, v8);
        v15 = (unsigned int)v20;
        if ( (unsigned int)v20 >= HIDWORD(v20) )
        {
          v17 = v14;
          sub_16CD150(&v19, v21, 0, 8);
          v15 = (unsigned int)v20;
          v14 = v17;
        }
        ++v13;
        *(_QWORD *)&v19[8 * v15] = v14;
        LODWORD(v20) = v20 + 1;
      }
      while ( (__int64 *)v12 != v13 );
    }
  }
  result = sub_1481AD0(a1, (__int64)&v19, a3, a4);
  if ( v19 != v21 )
  {
    v18 = result;
    _libc_free((unsigned __int64)v19);
    return v18;
  }
  return result;
}

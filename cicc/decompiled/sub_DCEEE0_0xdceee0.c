// Function: sub_DCEEE0
// Address: 0xdceee0
//
_QWORD *__fastcall sub_DCEEE0(__int64 *a1, __int64 a2, char a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 *v6; // r15
  __int64 *v7; // rbx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 *v12; // r9
  __int64 v13; // rbx
  __int64 *v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdx
  _QWORD *result; // rax
  _QWORD *v19; // [rsp+8h] [rbp-68h]
  _QWORD *v21; // [rsp+18h] [rbp-58h]
  _BYTE *v22; // [rsp+20h] [rbp-50h] BYREF
  __int64 v23; // [rsp+28h] [rbp-48h]
  _BYTE v24[64]; // [rsp+30h] [rbp-40h] BYREF

  v5 = *(unsigned int *)(a2 + 8);
  v6 = *(__int64 **)a2;
  if ( v5 == 1 )
    return (_QWORD *)*v6;
  v7 = &v6[v5];
  if ( v7 == v6 )
  {
    v22 = v24;
    v23 = 0x200000000LL;
  }
  else
  {
    v9 = 0;
    do
    {
      while ( 1 )
      {
        v11 = *v6;
        if ( !v9 )
          break;
        v10 = sub_D95540(v11);
        ++v6;
        v9 = sub_D970B0((__int64)a1, v9, v10);
        if ( v7 == v6 )
          goto LABEL_7;
      }
      ++v6;
      v9 = sub_D95540(v11);
    }
    while ( v7 != v6 );
LABEL_7:
    v12 = *(__int64 **)a2;
    v13 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
    v22 = v24;
    v23 = 0x200000000LL;
    if ( v12 != (__int64 *)v13 )
    {
      v14 = v12;
      do
      {
        v15 = sub_DC2CB0((__int64)a1, *v14, v9);
        v17 = (unsigned int)v23;
        if ( (unsigned __int64)(unsigned int)v23 + 1 > HIDWORD(v23) )
        {
          v19 = v15;
          sub_C8D5F0((__int64)&v22, v24, (unsigned int)v23 + 1LL, 8u, a5, v16);
          v17 = (unsigned int)v23;
          v15 = v19;
        }
        a4 = (__int64)v22;
        ++v14;
        *(_QWORD *)&v22[8 * v17] = v15;
        LODWORD(v23) = v23 + 1;
      }
      while ( (__int64 *)v13 != v14 );
    }
  }
  result = sub_DCEE50(a1, (__int64)&v22, a3, a4, a5);
  if ( v22 != v24 )
  {
    v21 = result;
    _libc_free(v22, &v22);
    return v21;
  }
  return result;
}

// Function: sub_3735B10
// Address: 0x3735b10
//
__int64 *__fastcall sub_3735B10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rax
  __int64 *result; // rax
  __int64 v11; // r14
  _QWORD *v12; // rax
  __int64 v13; // r14
  _QWORD *v14; // rax

  sub_322E2F0(*(_QWORD *)(a1 + 208), a2);
  v7 = *(_QWORD *)(a1 + 208);
  v8 = *(_QWORD *)(v7 + 3056);
  *(_QWORD *)(v7 + 3056) = a1;
  v9 = *(unsigned int *)(a1 + 480);
  if ( a1 == v8 && (_DWORD)v9 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 472) + 16 * v9 - 8);
    v12 = *(_QWORD **)v11;
    if ( !*(_QWORD *)v11 )
    {
      if ( (*(_BYTE *)(v11 + 9) & 0x70) != 0x20 || *(char *)(v11 + 8) < 0 )
        BUG();
      *(_BYTE *)(v11 + 8) |= 8u;
      v12 = sub_E807D0(*(_QWORD *)(v11 + 24));
      *(_QWORD *)v11 = v12;
    }
    v13 = v12[1];
    v14 = *(_QWORD **)a3;
    if ( !*(_QWORD *)a3 )
    {
      if ( (*(_BYTE *)(a3 + 9) & 0x70) != 0x20 || *(char *)(a3 + 8) < 0 )
        BUG();
      *(_BYTE *)(a3 + 8) |= 8u;
      v14 = sub_E807D0(*(_QWORD *)(a3 + 24));
      *(_QWORD *)a3 = v14;
    }
    if ( v14[1] == v13 )
    {
      result = (__int64 *)(*(_QWORD *)(a1 + 472) + 16LL * *(unsigned int *)(a1 + 480));
      *(result - 1) = a3;
      return result;
    }
    goto LABEL_7;
  }
  if ( v8 )
  {
LABEL_7:
    sub_3226DF0(*(_QWORD *)(a1 + 208), v8);
    v9 = *(unsigned int *)(a1 + 480);
  }
  if ( v9 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 484) )
  {
    sub_C8D5F0(a1 + 472, (const void *)(a1 + 488), v9 + 1, 0x10u, v5, v6);
    v9 = *(unsigned int *)(a1 + 480);
  }
  result = (__int64 *)(*(_QWORD *)(a1 + 472) + 16 * v9);
  *result = a2;
  result[1] = a3;
  ++*(_DWORD *)(a1 + 480);
  return result;
}

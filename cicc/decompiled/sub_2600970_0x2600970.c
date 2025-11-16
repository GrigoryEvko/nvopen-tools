// Function: sub_2600970
// Address: 0x2600970
//
unsigned __int64 __fastcall sub_2600970(__int64 a1, _QWORD **a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r13
  int v6; // edx
  int v7; // r14d
  unsigned __int64 v8; // rax
  signed __int64 v9; // rax
  int v10; // edx
  unsigned __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r13
  __int64 *v14; // rdx
  __int64 v15; // rax
  bool v16; // of
  __int64 v17; // rax
  __int64 v18; // rcx
  unsigned __int64 v19; // rax
  signed __int64 v20; // rax
  int v21; // edx
  unsigned __int64 result; // rax
  bool v23; // cc
  __int64 v24; // [rsp+0h] [rbp-40h]
  __int64 v25; // [rsp+8h] [rbp-38h]

  v4 = sub_25FCBE0(a1, (__int64 **)a3);
  v5 = v4;
  v7 = v6;
  if ( v6 == 1 )
    *(_DWORD *)(a3 + 288) = 1;
  v8 = *(_QWORD *)(a3 + 280) + v4;
  if ( __OFADD__(*(_QWORD *)(a3 + 280), v5) )
  {
    v8 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v5 <= 0 )
      v8 = 0x8000000000000000LL;
  }
  *(_QWORD *)(a3 + 280) = v8;
  v9 = sub_25FCC90(a1, (__int64 **)a3);
  if ( v10 == 1 )
    *(_DWORD *)(a3 + 304) = 1;
  if ( __OFADD__(*(_QWORD *)(a3 + 296), v9) )
  {
    v23 = v9 <= 0;
    v11 = 0x7FFFFFFFFFFFFFFFLL;
    if ( v23 )
      v11 = 0x8000000000000000LL;
  }
  else
  {
    v11 = *(_QWORD *)(a3 + 296) + v9;
  }
  *(_QWORD *)(a3 + 296) = v11;
  v24 = (__int64)(*(_QWORD *)(a3 + 8) - *(_QWORD *)a3) >> 3;
  v12 = v5 % v24;
  v13 = v5 / v24;
  v25 = (__int64)(*(_QWORD *)(a3 + 32) - *(_QWORD *)(a3 + 24)) >> 3;
  v14 = (__int64 *)(*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(a1 + 40))(
                     *(_QWORD *)(a1 + 48),
                     *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(***(_QWORD ***)a3 + 8LL) + 16LL) + 40LL) + 72LL),
                     v12);
  if ( v7 == 1 )
    *(_DWORD *)(a3 + 304) = 1;
  v15 = *(_QWORD *)(a3 + 296) + v13;
  if ( __OFADD__(*(_QWORD *)(a3 + 296), v13) )
  {
    if ( v13 <= 0 )
    {
      v19 = (unsigned int)v25 + 0x8000000000000000LL + (unsigned int)(2 * v25 * v24);
      goto LABEL_17;
    }
    v15 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v16 = __OFADD__((unsigned int)v25, v15);
  v17 = (unsigned int)v25 + v15;
  if ( v16 )
  {
    if ( !(_DWORD)v25 )
    {
      v19 = 0x8000000000000000LL;
      goto LABEL_17;
    }
    v17 = 0x7FFFFFFFFFFFFFFFLL;
  }
  v18 = (unsigned int)(2 * v24 * v25);
  v16 = __OFADD__(v18, v17);
  v19 = v18 + v17;
  if ( v16 )
  {
    v19 = 0x7FFFFFFFFFFFFFFFLL;
    if ( !(2 * (_DWORD)v24 * (_DWORD)v25) )
      v19 = 0x8000000000000000LL;
  }
LABEL_17:
  *(_QWORD *)(a3 + 296) = v19;
  v20 = sub_2600130(a2, a3, v14);
  if ( v21 == 1 )
    *(_DWORD *)(a3 + 304) = 1;
  if ( __OFADD__(*(_QWORD *)(a3 + 296), v20) )
  {
    v23 = v20 <= 0;
    result = 0x7FFFFFFFFFFFFFFFLL;
    if ( v23 )
      result = 0x8000000000000000LL;
  }
  else
  {
    result = *(_QWORD *)(a3 + 296) + v20;
  }
  *(_QWORD *)(a3 + 296) = result;
  return result;
}

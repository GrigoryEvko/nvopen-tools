// Function: sub_1D2EDB0
// Address: 0x1d2edb0
//
__int64 __fastcall sub_1D2EDB0(_QWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // r13
  unsigned __int8 v6; // al
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // rcx
  bool v10; // dl
  __int64 result; // rax
  __int64 v12; // rax
  unsigned __int8 v13; // cl
  unsigned __int64 v14; // rsi
  __int64 v15; // rdx
  bool v16; // al
  bool v17; // dl
  __int64 v18; // rdx
  __int64 v19; // rbx
  unsigned __int8 v20; // cl
  unsigned __int64 v21; // rdi
  unsigned __int8 v22; // dl
  bool v23; // al

  if ( (_QWORD *)a2 == a1 + 1 )
  {
    if ( !a1[5] )
      return sub_1D2ECF0((__int64)a1, (char *)a3);
    v19 = a1[4];
    v20 = *(_BYTE *)a3;
    v21 = *(_QWORD *)(a3 + 8);
    v22 = *(_BYTE *)(v19 + 32);
    v23 = v20 > v22;
    if ( v20 == v22 )
      v23 = v21 > *(_QWORD *)(v19 + 40);
    if ( !v23 )
      return sub_1D2ECF0((__int64)a1, (char *)a3);
    return 0;
  }
  v5 = *(_BYTE *)a3;
  v6 = *(_BYTE *)(a2 + 32);
  v8 = *(_QWORD *)(a3 + 8);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_BYTE *)a3 < v6;
  if ( v5 == v6 )
    v10 = v9 > v8;
  if ( v10 )
  {
    result = a2;
    if ( a1[3] == a2 )
      return result;
    v12 = sub_220EF80(a2);
    v13 = *(_BYTE *)(v12 + 32);
    v14 = *(_QWORD *)(v12 + 40);
    v15 = v12;
    v16 = v5 > v13;
    if ( v5 == v13 )
      v16 = v14 < v8;
    if ( v16 )
    {
      result = 0;
      if ( *(_QWORD *)(v15 + 24) )
        return a2;
      return result;
    }
    return sub_1D2ECF0((__int64)a1, (char *)a3);
  }
  v17 = v5 > v6;
  if ( v5 == v6 )
    v17 = v9 < v8;
  if ( !v17 )
    return a2;
  if ( a1[4] == a2 )
    return 0;
  v18 = sub_220EEE0(a2);
  if ( *(_BYTE *)(v18 + 32) == v5 )
  {
    if ( v8 >= *(_QWORD *)(v18 + 40) )
      return sub_1D2ECF0((__int64)a1, (char *)a3);
  }
  else if ( *(_BYTE *)(v18 + 32) <= v5 )
  {
    return sub_1D2ECF0((__int64)a1, (char *)a3);
  }
  result = 0;
  if ( *(_QWORD *)(a2 + 24) )
    return v18;
  return result;
}

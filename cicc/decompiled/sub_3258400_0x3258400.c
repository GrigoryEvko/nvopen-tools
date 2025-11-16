// Function: sub_3258400
// Address: 0x3258400
//
__int64 __fastcall sub_3258400(__int64 *a1)
{
  __int64 v1; // rbx
  __int64 v2; // rax
  __int64 v3; // r15
  __int64 v4; // r13
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 result; // rax
  __int64 *v8; // r13
  __int64 *j; // r12
  __int64 v10; // rsi
  __int64 i; // [rsp+0h] [rbp-40h]
  void (__fastcall *v12)(__int64, __int64); // [rsp+8h] [rbp-38h]

  v1 = *(_QWORD *)(a1[1] + 224);
  v2 = *(_QWORD *)(a1[2] + 2488);
  v3 = *(_QWORD *)(v2 + 32);
  v4 = v2 + 24;
  for ( i = v2; v4 != v3; v3 = *(_QWORD *)(v3 + 8) )
  {
    while ( 1 )
    {
      v5 = 0;
      if ( v3 )
        v5 = v3 - 56;
      if ( (unsigned __int8)sub_B2D620(v5, "safeseh", 7u) )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( v4 == v3 )
        goto LABEL_8;
    }
    v12 = *(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v1 + 344LL);
    v6 = sub_31DB510(a1[1], v5);
    v12(v1, v6);
  }
LABEL_8:
  result = sub_BA91D0(i, "ehcontguard", 0xBu);
  if ( result )
  {
    result = a1[6];
    if ( a1[7] != result )
    {
      result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v1 + 176LL))(
                 v1,
                 *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1[1] + 216) + 168LL) + 720LL),
                 0);
      v8 = (__int64 *)a1[7];
      for ( j = (__int64 *)a1[6];
            v8 != j;
            result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v1 + 352LL))(v1, v10) )
      {
        v10 = *j++;
      }
    }
  }
  return result;
}

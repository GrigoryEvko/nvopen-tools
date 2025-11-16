// Function: sub_C2BC00
// Address: 0xc2bc00
//
__int64 __fastcall sub_C2BC00(__int64 a1)
{
  _QWORD *v1; // rbx
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rsi
  int v8; // r12d
  _QWORD *i; // [rsp+10h] [rbp-50h]
  __int64 v10; // [rsp+20h] [rbp-40h] BYREF
  __int64 v11[7]; // [rsp+28h] [rbp-38h] BYREF

  v1 = *(_QWORD **)(a1 + 400);
  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  for ( i = *(_QWORD **)(a1 + 408); i != v1; v1 += 5 )
  {
    v5 = v1[3];
    if ( v5 )
    {
      v6 = v1[1];
      if ( !*(_BYTE *)(a1 + 205) || (v6 & 2) == 0 )
      {
        v7 = v2 + v1[2];
        v8 = v6 & 1;
        if ( v8 )
        {
          result = sub_C2BA80((_QWORD *)a1, v7, v1[3], (__int64)&v10, v11);
          if ( (_DWORD)result )
            return result;
          v7 = v10;
          v5 = v11[0];
        }
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD *))(*(_QWORD *)a1 + 96LL))(a1, v7, v5, v1);
        if ( (_DWORD)result )
          return result;
        if ( *(_QWORD *)(a1 + 208) != v5 + v7 )
        {
          sub_C1AFD0();
          return 5;
        }
        if ( v8 )
        {
          v4 = *(_QWORD *)(a1 + 72);
          *(_QWORD *)(a1 + 208) = v2 + v1[2];
          *(_QWORD *)(a1 + 216) = v2 + *(_QWORD *)(v4 + 16) - *(_QWORD *)(v4 + 8);
        }
      }
    }
  }
  sub_C1AFD0();
  return 0;
}

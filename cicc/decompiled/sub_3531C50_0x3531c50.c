// Function: sub_3531C50
// Address: 0x3531c50
//
__int64 __fastcall sub_3531C50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdx
  int *v5; // rcx
  int v6; // edx
  __int64 v7; // r13
  int *v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r14
  __int64 v11; // r12
  __int64 v12; // [rsp+8h] [rbp-58h]
  __int64 v13; // [rsp+10h] [rbp-50h]
  __int64 v14; // [rsp+20h] [rbp-40h]
  int *i; // [rsp+28h] [rbp-38h]

  result = *(_QWORD *)(a3 + 328);
  v12 = a3 + 320;
  v13 = result;
  if ( result != a3 + 320 )
  {
    while ( 1 )
    {
      v14 = *(_QWORD *)(v13 + 56);
      if ( v13 + 48 != v14 )
        break;
LABEL_17:
      result = *(_QWORD *)(v13 + 8);
      v13 = result;
      if ( v12 == result )
        return result;
    }
    while ( 1 )
    {
      v4 = *(_QWORD *)(v14 + 48);
      v5 = (int *)(v4 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v4 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v6 = v4 & 7;
        if ( !v6 )
        {
          v7 = 2;
          *(_QWORD *)(v14 + 48) = v5;
          v8 = (int *)(v14 + 48);
          goto LABEL_6;
        }
        if ( v6 == 3 )
        {
          v8 = v5 + 4;
          v7 = 2LL * *v5;
LABEL_6:
          for ( i = &v8[v7]; i != v8; v8 += 2 )
          {
            v9 = *(_QWORD *)(*(_QWORD *)v8 + 48LL);
            v10 = *(_QWORD *)(*(_QWORD *)v8 + 56LL);
            v11 = *(_QWORD *)(*(_QWORD *)v8 + 64LL);
            if ( *(_QWORD *)(*(_QWORD *)v8 + 40LL) )
              (*(void (__fastcall **)(__int64))(*(_QWORD *)a2 + 24LL))(a2);
            if ( v9 )
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 24LL))(a2, v9);
            if ( v10 )
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 24LL))(a2, v10);
            if ( v11 )
              (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a2 + 24LL))(a2, v11);
          }
        }
      }
      v14 = *(_QWORD *)(v14 + 8);
      if ( v13 + 48 == v14 )
        goto LABEL_17;
    }
  }
  return result;
}

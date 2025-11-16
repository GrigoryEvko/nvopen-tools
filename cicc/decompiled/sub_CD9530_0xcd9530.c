// Function: sub_CD9530
// Address: 0xcd9530
//
__int64 __fastcall sub_CD9530(__int64 a1, __int64 *a2)
{
  int v3; // ebx
  __int64 v4; // r13
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int8 *v10; // rsi
  __int64 v12; // [rsp+8h] [rbp-48h]
  __int64 v13[7]; // [rsp+18h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 64LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = a2[1] - *(_DWORD *)a2;
  if ( v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = v4 + 2;
    v7 = 1;
    v12 = v6;
    do
    {
      while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 72LL))(
                 a1,
                 (unsigned int)v5,
                 v13) )
      {
        ++v5;
        if ( ++v7 == v12 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
      }
      v8 = *a2;
      v9 = a2[1] - *a2;
      if ( v9 <= v5 )
      {
        if ( v9 < v7 )
        {
          sub_CD93F0(a2, v7 - v9);
          v8 = *a2;
        }
        else if ( v9 > v7 && a2[1] != v8 + v7 )
        {
          a2[1] = v8 + v7;
        }
      }
      v10 = (unsigned __int8 *)(v5 + v8);
      ++v5;
      ++v7;
      sub_CCCBF0(a1, v10);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 80LL))(a1, v13[0]);
    }
    while ( v7 != v12 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 88LL))(a1);
}

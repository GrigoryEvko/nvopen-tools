// Function: sub_18839E0
// Address: 0x18839e0
//
__int64 __fastcall sub_18839E0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v12; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]
  __int64 v14[7]; // [rsp+18h] [rbp-38h] BYREF

  LODWORD(v3) = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v3 = (a2[1] - *a2) >> 4;
  if ( (_DWORD)v3 )
  {
    v4 = (unsigned int)(v3 - 1);
    v5 = 0;
    v6 = v4 + 2;
    v7 = 1;
    v13 = v6;
    do
    {
      while ( !(*(unsigned __int8 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(
                 a1,
                 (unsigned int)v5,
                 v14) )
      {
        ++v5;
        if ( ++v7 == v13 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v8 = *a2;
      v9 = (a2[1] - *a2) >> 4;
      if ( v9 <= v5 )
      {
        if ( v9 < v7 )
        {
          sub_1883820((__int64)a2, v7 - v9);
          v8 = *a2;
        }
        else if ( v9 > v7 )
        {
          v12 = v8 + 16 * v7;
          if ( a2[1] != v12 )
            a2[1] = v12;
        }
      }
      v10 = v5++;
      ++v7;
      sub_187A340(a1, (_QWORD *)(16 * v10 + v8));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v14[0]);
    }
    while ( v7 != v13 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}

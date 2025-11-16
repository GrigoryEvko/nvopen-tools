// Function: sub_1156550
// Address: 0x1156550
//
__int64 __fastcall sub_1156550(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 v7; // r12
  __int64 v9; // rbx
  __int64 v10; // r13
  __int64 v11; // rdx
  unsigned int v12; // esi
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // rdx
  unsigned int v16; // esi
  _BYTE v18[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)a1[10] + 24LL))(
         a1[10],
         20,
         a2,
         a3,
         a5);
  if ( !v7 )
  {
    v19 = 257;
    if ( a5 )
    {
      v7 = sub_B504D0(20, a2, a3, (__int64)v18, 0, 0);
      sub_B448B0(v7, 1);
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v7,
        a4,
        a1[7],
        a1[8]);
      v13 = *a1;
      v14 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v14 )
      {
        do
        {
          v15 = *(_QWORD *)(v13 + 8);
          v16 = *(_DWORD *)v13;
          v13 += 16;
          sub_B99FD0(v7, v16, v15);
        }
        while ( v14 != v13 );
      }
    }
    else
    {
      v7 = sub_B504D0(20, a2, a3, (__int64)v18, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
        a1[11],
        v7,
        a4,
        a1[7],
        a1[8]);
      v9 = *a1;
      v10 = *a1 + 16LL * *((unsigned int *)a1 + 2);
      if ( *a1 != v10 )
      {
        do
        {
          v11 = *(_QWORD *)(v9 + 8);
          v12 = *(_DWORD *)v9;
          v9 += 16;
          sub_B99FD0(v7, v12, v11);
        }
        while ( v10 != v9 );
      }
    }
  }
  return v7;
}

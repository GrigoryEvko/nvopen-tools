// Function: sub_10A0530
// Address: 0x10a0530
//
unsigned __int8 *__fastcall sub_10A0530(__int64 *a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  __int64 v5; // rbx
  unsigned __int8 *v6; // r12
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // rdx
  unsigned int v11; // esi
  _BYTE v13[32]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v14; // [rsp+30h] [rbp-40h]

  v5 = sub_AD6530(*(_QWORD *)(a2 + 8), a2);
  v6 = (unsigned __int8 *)(*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a1[10] + 32LL))(
                            a1[10],
                            15,
                            v5,
                            a2,
                            0,
                            a4);
  if ( !v6 )
  {
    v14 = 257;
    v6 = (unsigned __int8 *)sub_B504D0(15, v5, a2, (__int64)v13, 0, 0);
    (*(void (__fastcall **)(__int64, unsigned __int8 *, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
      a1[11],
      v6,
      a3,
      a1[7],
      a1[8]);
    v8 = *a1;
    v9 = *a1 + 16LL * *((unsigned int *)a1 + 2);
    if ( *a1 != v9 )
    {
      do
      {
        v10 = *(_QWORD *)(v8 + 8);
        v11 = *(_DWORD *)v8;
        v8 += 16;
        sub_B99FD0((__int64)v6, v11, v10);
      }
      while ( v9 != v8 );
    }
    if ( a4 )
      sub_B44850(v6, 1);
  }
  return v6;
}

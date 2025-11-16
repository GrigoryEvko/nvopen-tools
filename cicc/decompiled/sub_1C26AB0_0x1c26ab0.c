// Function: sub_1C26AB0
// Address: 0x1c26ab0
//
__int64 __fastcall sub_1C26AB0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rax
  __int64 (__fastcall *v4)(); // rax
  unsigned int v5; // r12d
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rsi
  int v10; // eax
  __int64 v11; // r13
  __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rbx
  __int64 v16; // rax
  __int64 v17; // [rsp+0h] [rbp-30h] BYREF
  int v18; // [rsp+8h] [rbp-28h]

  v3 = *(_QWORD *)a1;
  v17 = 0;
  v18 = 0;
  v4 = *(__int64 (__fastcall **)())(v3 + 32);
  if ( v4 == sub_1C26BC0 )
  {
    sub_1C268F0(a1);
    v5 = 0;
    sub_1C26A00(a1, (__int64)sub_1C26BC0, v6, v7, v8);
    if ( !*(_DWORD *)(a1 + 72) )
    {
      v16 = sub_163A1D0();
      sub_1705E40(v16);
    }
  }
  else
  {
    v5 = ((__int64 (__fastcall *)(__int64, __int64 *))v4)(a1, &v17);
  }
  v9 = v17;
  if ( v17 )
  {
    v10 = v18;
    if ( v18 )
    {
      v11 = (unsigned int)(v18 - 1);
      *a2 = *(_QWORD *)(v17 + 8 * v11);
      v12 = *(_QWORD *)(a1 + 24);
      if ( v10 != 1 )
      {
        v13 = 8 * v11;
        v14 = 0;
        do
        {
          if ( *(_QWORD *)(v9 + v14) )
          {
            (*(void (__fastcall **)(_QWORD, _QWORD))(v12 + 8))(*(_QWORD *)(v12 + 16), *(_QWORD *)(v9 + v14));
            v9 = v17;
          }
          v14 += 8;
        }
        while ( v14 != v13 );
      }
    }
    else
    {
      *a2 = 0;
      v12 = *(_QWORD *)(a1 + 24);
    }
    (*(void (__fastcall **)(_QWORD))(v12 + 8))(*(_QWORD *)(v12 + 16));
  }
  else
  {
    *a2 = 0;
  }
  return v5;
}

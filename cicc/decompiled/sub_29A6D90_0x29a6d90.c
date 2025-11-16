// Function: sub_29A6D90
// Address: 0x29a6d90
//
__int64 __fastcall sub_29A6D90(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rax
  __int64 v4; // r13
  int v6; // eax
  __int64 v7; // rax
  _QWORD *v8; // r15
  __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  _BYTE *v15; // [rsp+18h] [rbp-48h]
  _BYTE *v16; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)a1 == 1 )
  {
    v3 = sub_29A6D90(*(_QWORD *)(a1 - 32), a2);
    v4 = v3;
    if ( *(_QWORD *)(a1 - 32) != v3 )
    {
      sub_B303B0(a1, v3);
      *a2 = 1;
    }
  }
  else
  {
    v4 = a1;
    if ( *(_BYTE *)a1 == 5 )
    {
      v6 = *(_DWORD *)(a1 + 4);
      v14 = 0;
      v15 = 0;
      v16 = 0;
      v7 = 4LL * (v6 & 0x7FFFFFF);
      if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      {
        v8 = *(_QWORD **)(a1 - 8);
        v4 = (__int64)&v8[v7];
      }
      else
      {
        v8 = (_QWORD *)(a1 - v7 * 8);
      }
      if ( v8 == (_QWORD *)v4 )
      {
        v12 = 0;
        v11 = 0;
      }
      else
      {
        do
        {
          v9 = sub_29A6D90(*v8, a2);
          v10 = v15;
          v13 = v9;
          if ( v15 == v16 )
          {
            sub_262AD50((__int64)&v14, v15, &v13);
          }
          else
          {
            if ( v15 )
            {
              *(_QWORD *)v15 = v9;
              v10 = v15;
            }
            v15 = v10 + 8;
          }
          v8 += 4;
        }
        while ( (_QWORD *)v4 != v8 );
        v11 = v14;
        v12 = (__int64)&v15[-v14] >> 3;
      }
      v4 = sub_ADABF0(a1, v11, v12, *(__int64 ***)(a1 + 8), 0, 0);
      if ( v14 )
        j_j___libc_free_0(v14);
    }
  }
  return v4;
}

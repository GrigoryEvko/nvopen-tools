// Function: sub_2739FD0
// Address: 0x2739fd0
//
__int64 __fastcall sub_2739FD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // r14
  __int64 v6; // r12
  unsigned int v7; // eax
  __int64 v8; // rbx
  unsigned __int64 v10; // r8
  int v11; // ecx
  unsigned __int64 v12; // rsi
  int v13; // eax
  bool v14; // al

  v3 = a2 - a1;
  v5 = a1;
  v6 = v3 >> 6;
  if ( v3 > 0 )
  {
    do
    {
      v7 = *(_DWORD *)(a3 + 48);
      v8 = v5 + (v6 >> 1 << 6);
      if ( *(_DWORD *)(v8 + 48) == v7 )
      {
        v13 = *(_DWORD *)(v8 + 56);
        v11 = *(_DWORD *)(a3 + 56);
        if ( v13 )
        {
          if ( !v11 )
            goto LABEL_13;
          v10 = *(_QWORD *)v8;
          if ( v13 == 3 )
            v10 = sub_2739680(*(_QWORD *)v8);
          v12 = *(_QWORD *)a3;
          if ( v11 == 3 )
            v12 = sub_2739680(*(_QWORD *)a3);
          if ( !sub_B445A0(v10, v12) )
          {
LABEL_13:
            v6 >>= 1;
            continue;
          }
        }
        else if ( !v11 )
        {
          v14 = 0;
          if ( **(_BYTE **)(v8 + 8) != 17 )
            v14 = **(_BYTE **)(v8 + 16) != 17;
          if ( **(_BYTE **)(a3 + 8) == 17 || (unsigned __int8)(**(_BYTE **)(a3 + 16) != 17) <= (unsigned __int8)v14 )
            goto LABEL_13;
        }
      }
      else if ( *(_DWORD *)(v8 + 48) >= v7 )
      {
        goto LABEL_13;
      }
      v5 = v8 + 64;
      v6 = v6 - (v6 >> 1) - 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}

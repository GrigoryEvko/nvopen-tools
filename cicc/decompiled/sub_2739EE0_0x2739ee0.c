// Function: sub_2739EE0
// Address: 0x2739ee0
//
__int64 __fastcall sub_2739EE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rbx
  unsigned int v8; // eax
  unsigned __int64 v10; // r8
  int v11; // ecx
  unsigned __int64 v12; // rsi
  int v13; // eax

  v3 = a2 - a1;
  v5 = a1;
  v6 = v3 >> 6;
  if ( v3 > 0 )
  {
    do
    {
      v7 = v5 + (v6 >> 1 << 6);
      v8 = *(_DWORD *)(v7 + 48);
      if ( *(_DWORD *)(a3 + 48) == v8 )
      {
        v13 = *(_DWORD *)(a3 + 56);
        v11 = *(_DWORD *)(v7 + 56);
        if ( v13 )
        {
          if ( !v11 )
            goto LABEL_13;
          v10 = *(_QWORD *)a3;
          if ( v13 == 3 )
            v10 = sub_2739680(*(_QWORD *)a3);
          v12 = *(_QWORD *)v7;
          if ( v11 == 3 )
            v12 = sub_2739680(*(_QWORD *)v7);
          if ( !sub_B445A0(v10, v12) )
          {
LABEL_13:
            v5 = v7 + 64;
            v6 = v6 - (v6 >> 1) - 1;
            continue;
          }
        }
        else if ( !v11 )
        {
          if ( **(_BYTE **)(a3 + 8) != 17 )
            LOBYTE(v13) = **(_BYTE **)(a3 + 16) != 17;
          if ( **(_BYTE **)(v7 + 8) == 17 || (unsigned __int8)(**(_BYTE **)(v7 + 16) != 17) <= (unsigned __int8)v13 )
            goto LABEL_13;
        }
      }
      else if ( *(_DWORD *)(a3 + 48) >= v8 )
      {
        goto LABEL_13;
      }
      v6 >>= 1;
    }
    while ( v6 > 0 );
  }
  return v5;
}

// Function: sub_2778F20
// Address: 0x2778f20
//
__int64 __fastcall sub_2778F20(unsigned __int8 *a1, __int64 a2)
{
  unsigned int v2; // r14d
  int v3; // edx
  __int64 v5; // rbx
  __int64 v6; // r13
  unsigned __int8 *v7; // rsi
  unsigned __int8 *v8; // r15
  int v9; // edx
  int v10; // eax
  int v11; // eax
  int v12; // [rsp+4h] [rbp-4Ch]
  unsigned __int8 v13; // [rsp+4h] [rbp-4Ch]
  int v14; // [rsp+10h] [rbp-40h]
  __int64 v15; // [rsp+18h] [rbp-38h]

  if ( a1 == (unsigned __int8 *)a2 )
    return 1;
  v2 = 0;
  v3 = *a1;
  if ( (unsigned int)(v3 - 12) > 1 )
  {
    LOBYTE(v2) = (_BYTE)v3 != 11 || *(_BYTE *)a2 != 11;
    if ( (_BYTE)v2 )
      return 0;
    if ( *((_QWORD *)a1 + 1) != *(_QWORD *)(a2 + 8) )
      return v2;
    if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) != 0 )
    {
      v5 = 0;
      v15 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      v6 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      do
      {
        v7 = *(unsigned __int8 **)&a1[v5 - v15];
        v8 = *(unsigned __int8 **)(v6 + v5);
        v9 = *v7;
        if ( (_BYTE)v9 == 17 )
        {
          if ( *((_DWORD *)v7 + 8) <= 0x40u )
          {
            if ( !*((_QWORD *)v7 + 3) )
              goto LABEL_18;
          }
          else
          {
            v12 = *((_DWORD *)v7 + 8);
            v10 = sub_C444A0((__int64)(v7 + 24));
            v9 = 17;
            if ( v12 == v10 )
              goto LABEL_18;
          }
          v11 = *v8;
          if ( (_BYTE)v11 != 17 )
            goto LABEL_17;
        }
        else
        {
          v11 = *v8;
          if ( (_BYTE)v11 != 17 )
            goto LABEL_16;
        }
        if ( *((_DWORD *)v8 + 8) <= 0x40u )
        {
          v11 = 17;
          if ( !*((_QWORD *)v8 + 3) )
            goto LABEL_16;
        }
        else
        {
          v14 = *((_DWORD *)v8 + 8);
          v13 = v9;
          if ( v14 == (unsigned int)sub_C444A0((__int64)(v8 + 24)) )
          {
            v9 = v13;
            v11 = 17;
LABEL_16:
            if ( (unsigned int)(v9 - 12) <= 1 )
              return v2;
LABEL_17:
            if ( v7 != v8 || (unsigned int)(v11 - 12) <= 1 )
              return 0;
          }
        }
LABEL_18:
        v5 += 32;
      }
      while ( v15 != v5 );
    }
    return 1;
  }
  return v2;
}

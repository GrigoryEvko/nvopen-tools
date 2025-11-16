// Function: sub_7605A0
// Address: 0x7605a0
//
void __fastcall sub_7605A0(__int64 a1)
{
  __int64 i; // rbx
  char v2; // al
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 *v5; // rax
  __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // [rsp-40h] [rbp-40h]

  if ( (*(_BYTE *)(a1 + 206) & 0x10) == 0 )
  {
    for ( i = a1; ; i = v6 )
    {
      if ( !dword_4F08010 || (*(_BYTE *)(i - 8) & 2) != 0 )
      {
        v2 = *(_BYTE *)(i + 203);
        if ( (v2 & 4) != 0 )
          return;
        *(_BYTE *)(i + 203) = v2 | 4;
        if ( (*(_BYTE *)(i + 193) & 0x20) != 0 )
        {
          if ( *(_DWORD *)(i + 160) )
          {
            v7 = dword_4F07270[0];
            v8 = sub_72B840(i);
            v9 = v8;
            if ( (*(_BYTE *)(v8 + 29) & 1) != 0 )
            {
              v10 = v8;
              dword_4F07270[0] = *(_DWORD *)(i + 164);
              v11 = qword_4F04C50;
              qword_4F04C50 = v9;
              v12 = v11;
              sub_7604D0(v10, 0x17u);
              dword_4F07270[0] = v7;
              qword_4F04C50 = v12;
              sub_75BCD0(i);
              if ( *(_DWORD *)(v9 + 240) == -1
                && v9 != qword_4F04C50
                && (!qword_4F04C50 || *(_DWORD *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 164LL) != dword_4F07270[0]) )
              {
                sub_823780(*(unsigned int *)(i + 164));
              }
            }
          }
        }
        if ( *(_BYTE *)(i + 174) == 6 )
          sub_7605A0(*(_QWORD *)(i + 176));
        v3 = *(_QWORD *)(i + 272);
        if ( v3 )
          sub_7605A0(v3);
        v4 = *(_QWORD *)(i + 320);
        if ( v4 )
          sub_7605A0(v4);
      }
      v5 = *(__int64 **)(i + 32);
      if ( !v5 )
        break;
      v6 = *v5;
      if ( v6 == i || (*(_BYTE *)(v6 - 8) & 2) == 0 || (*(_BYTE *)(v6 + 206) & 0x10) != 0 )
        break;
    }
  }
}

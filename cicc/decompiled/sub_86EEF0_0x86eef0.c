// Function: sub_86EEF0
// Address: 0x86eef0
//
void __fastcall sub_86EEF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  _BYTE *v3; // rax
  __int64 v4; // r13
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rsi

  v2 = a2;
  if ( a1 )
  {
    *(_BYTE *)(a1 + 120) = qword_4F5FD78 & 1 | *(_BYTE *)(a1 + 120) & 0xFE;
    v3 = sub_86E480(7u, (unsigned int *)(a1 + 64));
    *(_QWORD *)(a1 + 128) = v3;
    *((_QWORD *)v3 + 9) = a1;
  }
  if ( dword_4F077C4 == 2 || dword_4D047EC )
  {
    v4 = sub_86B2C0(3);
    *(_QWORD *)(v4 + 40) = *(_QWORD *)(a1 + 128);
    *(_QWORD *)(v4 + 24) = *(_QWORD *)&dword_4F063F8;
    sub_86CBE0(v4);
    if ( dword_4F077C4 == 2 )
    {
      v5 = sub_7340A0(qword_4F06BC0);
      *(_QWORD *)(*(_QWORD *)(a1 + 128) + 80LL) = v5;
      if ( a2 )
      {
        do
        {
          v6 = *(_QWORD *)(v2 + 40);
          v7 = v5;
          v8 = *(_QWORD *)(v6 + 80);
          if ( v8 != v5 )
            v7 = sub_86BCA0(v5, v8);
          *(_QWORD *)(v6 + 80) = v7;
          v2 = *(_QWORD *)(v2 + 48);
        }
        while ( v2 );
      }
    }
    else if ( dword_4D047EC && unk_4D047E8 && (*(_BYTE *)(a1 + 120) & 0xA) != 0 )
    {
      while ( v2 )
      {
        sub_86C360(v2, v4);
        v2 = *(_QWORD *)(v2 + 48);
      }
    }
  }
}

// Function: sub_E8F2F0
// Address: 0xe8f2f0
//
void __fastcall sub_E8F2F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  unsigned __int64 v4; // r10
  __int64 i; // r8
  unsigned __int64 v6; // rcx
  __int64 v7; // rdi
  __int64 v8; // r11
  unsigned int v9; // esi
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // rdx

  if ( a1 != a2 )
  {
    v2 = a1;
    if ( a2 != a1 + 24 )
    {
      v4 = 0xAAAAAAAAAAAAAAABLL;
      for ( i = a1 + 48; ; i += 24 )
      {
        v6 = *(_QWORD *)(i - 16);
        v7 = i - 24;
        v8 = i;
        if ( v6 < *(_QWORD *)(v2 + 8) )
          break;
        if ( v6 == *(_QWORD *)(v2 + 8) )
        {
          v9 = *(_DWORD *)(i - 24);
          if ( v9 < *(_DWORD *)v2 )
            goto LABEL_6;
        }
        sub_E8F2A0((unsigned int *)v7);
LABEL_9:
        if ( a2 == v8 )
          return;
      }
      v9 = *(_DWORD *)(i - 24);
LABEL_6:
      v10 = *(_QWORD *)(i - 8);
      v11 = v4 * ((v7 - v2) >> 3);
      if ( v7 - v2 > 0 )
      {
        do
        {
          v12 = *(_QWORD *)(v7 - 16);
          v7 -= 24;
          *(_QWORD *)(v7 + 32) = v12;
          *(_DWORD *)(v7 + 24) = *(_DWORD *)v7;
          *(_QWORD *)(v7 + 40) = *(_QWORD *)(v7 + 16);
          --v11;
        }
        while ( v11 );
      }
      *(_QWORD *)(v2 + 8) = v6;
      *(_DWORD *)v2 = v9;
      *(_QWORD *)(v2 + 16) = v10;
      goto LABEL_9;
    }
  }
}

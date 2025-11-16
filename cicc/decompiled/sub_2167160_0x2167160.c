// Function: sub_2167160
// Address: 0x2167160
//
__int64 __fastcall sub_2167160(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  char v4; // r14
  unsigned int v5; // r15d
  int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // rdx
  int v11; // [rsp+Ch] [rbp-34h]

  v11 = *(_QWORD *)(a2 + 32);
  if ( v11 > 0 )
  {
    v4 = a4;
    v5 = 0;
    v6 = 0;
    while ( 1 )
    {
      if ( a3 )
      {
        v7 = a2;
        if ( *(_BYTE *)(a2 + 8) == 16 )
          v7 = **(_QWORD **)(a2 + 16);
        v5 += sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v7, a4);
        if ( v4 )
          goto LABEL_9;
LABEL_4:
        if ( v11 == ++v6 )
          return v5;
      }
      else
      {
        if ( !v4 )
          goto LABEL_4;
LABEL_9:
        v8 = a2;
        if ( *(_BYTE *)(a2 + 8) == 16 )
          v8 = **(_QWORD **)(a2 + 16);
        ++v6;
        v5 += sub_1F43D80(*(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 8), v8, a4);
        if ( v11 == v6 )
          return v5;
      }
    }
  }
  return 0;
}

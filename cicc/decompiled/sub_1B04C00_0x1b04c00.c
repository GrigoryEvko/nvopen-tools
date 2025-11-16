// Function: sub_1B04C00
// Address: 0x1b04c00
//
__int64 __fastcall sub_1B04C00(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // r11
  __int64 v8; // rcx
  char v9; // si
  __int64 v10; // r10
  __int64 v11; // rax
  unsigned int v12; // r9d
  __int64 v13; // rdx
  __int64 v14; // rdx

  result = sub_157F280(a1);
  if ( result != v6 )
  {
    v7 = v6;
    v8 = result;
    do
    {
      v9 = *(_BYTE *)(v8 + 23) & 0x40;
      v10 = 24LL * *(unsigned int *)(v8 + 56);
      v11 = v10 + 8;
      v12 = *(_DWORD *)(v8 + 20) & 0xFFFFFFF;
      if ( v12 )
      {
        while ( 1 )
        {
          v13 = v8 - 24LL * v12;
          if ( v9 )
            v13 = *(_QWORD *)(v8 - 8);
          if ( a2 == *(_QWORD *)(v13 + v11) )
            break;
          v11 += 8;
          if ( v10 + 8LL * (v12 - 1) + 16 == v11 )
            goto LABEL_16;
        }
        if ( !v9 )
        {
LABEL_17:
          v14 = v8 - 24LL * v12;
          goto LABEL_11;
        }
      }
      else
      {
LABEL_16:
        v11 = v10 + 0x800000000LL;
        if ( !v9 )
          goto LABEL_17;
      }
      v14 = *(_QWORD *)(v8 - 8);
LABEL_11:
      *(_QWORD *)(v14 + v11) = a3;
      result = *(_QWORD *)(v8 + 32);
      if ( !result )
        BUG();
      v8 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v8 = result - 24;
    }
    while ( v7 != v8 );
  }
  return result;
}

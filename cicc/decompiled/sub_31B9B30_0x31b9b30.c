// Function: sub_31B9B30
// Address: 0x31b9b30
//
__int64 __fastcall sub_31B9B30(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // edx
  int v14; // r10d

  v6 = *(_QWORD *)(a2 + 8);
  if ( a3 )
    goto LABEL_9;
  while ( 1 )
  {
    v6 = sub_318B520(v6);
LABEL_9:
    if ( !v6 )
      return 0;
    v11 = *(unsigned int *)(a1 + 24);
    v12 = *(_QWORD *)(a1 + 8);
    if ( !(_DWORD)v11 )
      return 0;
    v7 = (v11 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v8 = (__int64 *)(v12 + 16LL * v7);
    v9 = *v8;
    if ( *v8 != v6 )
    {
      v13 = 1;
      while ( v9 != -4096 )
      {
        v14 = v13 + 1;
        v7 = (v11 - 1) & (v13 + v7);
        v8 = (__int64 *)(v12 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == v6 )
          goto LABEL_4;
        v13 = v14;
      }
      return 0;
    }
LABEL_4:
    if ( v8 == (__int64 *)(v12 + 16 * v11) )
      return 0;
    result = v8[1];
    if ( !result )
      return 0;
    if ( *(_DWORD *)(result + 16) == 1 && a4 != result )
      return result;
  }
}

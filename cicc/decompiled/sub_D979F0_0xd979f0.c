// Function: sub_D979F0
// Address: 0xd979f0
//
__int64 __fastcall sub_D979F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  int v8; // eax
  int v9; // edi
  unsigned int v10; // ecx
  __int64 *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 result; // rax
  __int64 v15; // rax
  int v16; // eax
  int v17; // r9d

  v5 = *(_QWORD *)(a3 + 40);
  if ( *(_QWORD *)(a2 + 40) == v5 )
  {
    result = sub_98CF00(a2 + 24, 0, a3 + 24, 0, 32);
    if ( (_BYTE)result )
      return result;
    v5 = *(_QWORD *)(a3 + 40);
  }
  v6 = *(_QWORD *)(a1 + 48);
  v7 = *(_QWORD *)(v6 + 8);
  v8 = *(_DWORD *)(v6 + 24);
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = (v8 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
    v11 = (__int64 *)(v7 + 16LL * v10);
    v12 = *v11;
    if ( *v11 == v5 )
    {
LABEL_4:
      v13 = v11[1];
      if ( v13 )
      {
        if ( **(_QWORD **)(v13 + 32) == v5 )
        {
          v15 = sub_D4B130(v13);
          if ( v15 == *(_QWORD *)(a2 + 40) )
          {
            if ( (unsigned __int8)sub_98CF00(a2 + 24, 0, v15 + 48, 0, 32) )
              return sub_98CF00(*(_QWORD *)(*(_QWORD *)(a3 + 40) + 56LL), 1, a3 + 24, 0, 32);
          }
        }
      }
    }
    else
    {
      v16 = 1;
      while ( v12 != -4096 )
      {
        v17 = v16 + 1;
        v10 = v9 & (v16 + v10);
        v11 = (__int64 *)(v7 + 16LL * v10);
        v12 = *v11;
        if ( *v11 == v5 )
          goto LABEL_4;
        v16 = v17;
      }
    }
  }
  return 0;
}

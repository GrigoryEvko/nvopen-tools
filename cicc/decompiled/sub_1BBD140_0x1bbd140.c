// Function: sub_1BBD140
// Address: 0x1bbd140
//
__int64 __fastcall sub_1BBD140(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  char v5; // bl
  __int64 v6; // r15
  int v7; // r12d
  int v8; // edi
  _QWORD *v9; // rcx
  unsigned int v10; // eax
  _QWORD *v11; // rdx
  int v12; // eax
  __int64 v13; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 1;
  result = 1;
  if ( *(_QWORD *)(v2 + 8) )
  {
    v13 = a1 + 40;
    v5 = *(_BYTE *)(a1 + 32) & 1;
    while ( 1 )
    {
      if ( v5 )
      {
        v6 = v13;
        v7 = 3;
      }
      else
      {
        v12 = *(_DWORD *)(a1 + 48);
        v6 = *(_QWORD *)(a1 + 40);
        if ( !v12 )
          return 0;
        v7 = v12 - 1;
      }
      v8 = 1;
      v9 = sub_1648700(v2);
      v10 = v7 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v11 = *(_QWORD **)(v6 + 16LL * v10);
      if ( v9 != v11 )
        break;
LABEL_6:
      v2 = *(_QWORD *)(v2 + 8);
      if ( !v2 )
        return 1;
    }
    while ( v11 != (_QWORD *)-8LL )
    {
      v10 = v7 & (v8 + v10);
      v11 = *(_QWORD **)(v6 + 16LL * v10);
      if ( v9 == v11 )
        goto LABEL_6;
      ++v8;
    }
    return 0;
  }
  return result;
}

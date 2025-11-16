// Function: sub_1377D90
// Address: 0x1377d90
//
__int64 __fastcall sub_1377D90(__int64 a1)
{
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // rcx
  int v5; // edx
  __int64 v6; // rdi
  __int64 v7; // rbx
  __int64 v8; // rsi
  int v9; // edx
  int v10; // r8d
  _QWORD v11[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v12; // [rsp+18h] [rbp-38h]
  __int64 v13; // [rsp+20h] [rbp-30h]

  sub_1377B70(*(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24));
  v2 = *(_QWORD *)(a1 + 32);
  result = *(unsigned int *)(v2 + 24);
  if ( (_DWORD)result )
  {
    v4 = *(_QWORD *)(a1 + 24);
    v5 = result - 1;
    v6 = *(_QWORD *)(v2 + 8);
    result = ((_DWORD)result - 1) & (((unsigned int)*(_QWORD *)(a1 + 24) >> 9) ^ ((unsigned int)v4 >> 4));
    v7 = v6 + 40 * result;
    v8 = *(_QWORD *)(v7 + 24);
    if ( v4 == v8 )
    {
LABEL_4:
      v12 = -16;
      v13 = 0;
      v11[0] = 2;
      result = *(_QWORD *)(v7 + 24);
      v11[1] = 0;
      if ( result == -16 )
      {
        *(_QWORD *)(v7 + 32) = 0;
      }
      else if ( result == -8 || !result )
      {
        *(_QWORD *)(v7 + 24) = -16;
        v9 = v12;
        LOBYTE(result) = v12 != 0;
        LOBYTE(v8) = v12 != -8;
        LOBYTE(v9) = v12 != -16;
        result = v9 & (unsigned int)v8 & (unsigned int)result;
        *(_QWORD *)(v7 + 32) = v13;
        if ( (_BYTE)result )
          result = sub_1649B30(v11);
      }
      else
      {
        sub_1649B30(v7 + 8);
        *(_QWORD *)(v7 + 24) = v12;
        result = v13;
        *(_QWORD *)(v7 + 32) = v13;
      }
      --*(_DWORD *)(v2 + 16);
      ++*(_DWORD *)(v2 + 20);
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        result = v5 & (unsigned int)(v10 + result);
        v7 = v6 + 40LL * (unsigned int)result;
        v8 = *(_QWORD *)(v7 + 24);
        if ( v4 == v8 )
          goto LABEL_4;
        ++v10;
      }
    }
  }
  return result;
}

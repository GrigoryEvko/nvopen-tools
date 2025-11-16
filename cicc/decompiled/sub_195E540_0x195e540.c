// Function: sub_195E540
// Address: 0x195e540
//
__int64 __fastcall sub_195E540(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned int v4; // r13d
  int v5; // edx
  __int64 v7; // rdi
  unsigned int v8; // esi
  __int64 v9; // rcx
  __int64 v10; // r14
  int v11; // edx
  __int64 v12; // rsi
  unsigned int v13; // r13d
  __int64 v14; // rcx
  unsigned int v15; // r8d
  unsigned int v16; // edi

  result = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)result )
  {
    v4 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
    v5 = result - 1;
    v7 = *(_QWORD *)(a1 + 168);
    v8 = (result - 1) & v4;
    result = v7 + 16LL * v8;
    v9 = *(_QWORD *)result;
    if ( *(_QWORD *)result == a2 )
    {
LABEL_3:
      v10 = *(_QWORD *)(result + 8);
      if ( v10 )
      {
        sub_12D5E00(*(_QWORD *)(result + 8));
        j_j___libc_free_0(v10, 72);
        result = *(unsigned int *)(a1 + 184);
        if ( (_DWORD)result )
        {
          v11 = result - 1;
          v12 = *(_QWORD *)(a1 + 168);
          v13 = (result - 1) & v4;
          result = v12 + 16LL * v13;
          v14 = *(_QWORD *)result;
          if ( *(_QWORD *)result == a2 )
          {
LABEL_6:
            *(_QWORD *)result = -16;
            --*(_DWORD *)(a1 + 176);
            ++*(_DWORD *)(a1 + 180);
          }
          else
          {
            result = 1;
            while ( v14 != -8 )
            {
              v16 = result + 1;
              v13 = v11 & (result + v13);
              result = v12 + 16LL * v13;
              v14 = *(_QWORD *)result;
              if ( *(_QWORD *)result == a2 )
                goto LABEL_6;
              result = v16;
            }
          }
        }
      }
    }
    else
    {
      result = 1;
      while ( v9 != -8 )
      {
        v15 = result + 1;
        v8 = v5 & (result + v8);
        result = v7 + 16LL * v8;
        v9 = *(_QWORD *)result;
        if ( *(_QWORD *)result == a2 )
          goto LABEL_3;
        result = v15;
      }
    }
  }
  return result;
}

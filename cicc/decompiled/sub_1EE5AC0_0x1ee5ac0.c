// Function: sub_1EE5AC0
// Address: 0x1ee5ac0
//
__int64 __fastcall sub_1EE5AC0(__int64 a1, unsigned int a2, __int64 a3)
{
  _QWORD *v4; // r13
  __int64 (*v5)(); // rax
  __int64 result; // rax
  __int64 v7; // rdx
  unsigned int v8; // eax
  __int16 v9; // r15
  unsigned __int16 *v10; // rdx
  unsigned __int16 v11; // r15
  unsigned __int16 *v12; // rbx
  __int64 v13; // rsi
  _DWORD *v14; // rax
  unsigned int v15; // r8d
  int v16; // r9d

  if ( (a2 & 0x80000000) != 0 )
    return (__int64)sub_1EE58A0(a3, a2 | 0xFFFFFFFF00000000LL);
  v4 = *(_QWORD **)(a1 + 16);
  v5 = *(__int64 (**)())(**(_QWORD **)(*v4 + 16LL) + 112LL);
  if ( v5 == sub_1D00B10 )
    BUG();
  result = *(_QWORD *)(v5() + 232);
  if ( *(_BYTE *)(result + 8LL * a2 + 4) )
  {
    result = *(_QWORD *)(v4[38] + 8LL * (a2 >> 6)) & (1LL << a2);
    if ( !result )
    {
      v7 = *(_QWORD *)(a1 + 8);
      v8 = *(_DWORD *)(*(_QWORD *)(v7 + 8) + 24LL * a2 + 16);
      v9 = a2 * (v8 & 0xF);
      result = *(_QWORD *)(v7 + 56) + 2LL * (v8 >> 4);
      v10 = (unsigned __int16 *)(result + 2);
      v11 = *(_WORD *)result + v9;
      while ( 1 )
      {
        v12 = v10;
        if ( !v10 )
          break;
        while ( 1 )
        {
          v13 = *(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8);
          v14 = sub_1EE52A0(*(_DWORD **)a3, v13, v11);
          if ( (_DWORD *)v13 == v14 )
          {
            if ( v15 >= *(_DWORD *)(a3 + 12) )
            {
              sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v15, v16);
              v14 = (_DWORD *)(*(_QWORD *)a3 + 8LL * *(unsigned int *)(a3 + 8));
            }
            *(_QWORD *)v14 = v11 | 0xFFFFFFFF00000000LL;
            ++*(_DWORD *)(a3 + 8);
          }
          else
          {
            v14[1] = -1;
          }
          result = *v12;
          v10 = 0;
          ++v12;
          if ( !(_WORD)result )
            break;
          v11 += result;
          if ( !v12 )
            return result;
        }
      }
    }
  }
  return result;
}

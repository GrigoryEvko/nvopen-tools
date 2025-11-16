// Function: sub_1EE5920
// Address: 0x1ee5920
//
__int64 __fastcall sub_1EE5920(__int64 a1, unsigned int a2, unsigned int a3, __int64 a4)
{
  __int64 v5; // r12
  __int64 v6; // rsi
  __int64 result; // rax
  _QWORD *v8; // r13
  __int64 (*v9)(); // rax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int16 v12; // r15
  unsigned __int16 *v13; // rdx
  unsigned __int16 v14; // r15
  unsigned __int16 *v15; // rbx
  __int64 v16; // rsi
  _DWORD *v17; // rax
  unsigned int v18; // r8d
  int v19; // r9d

  v5 = a2;
  if ( (a2 & 0x80000000) == 0 )
  {
    v8 = *(_QWORD **)(a1 + 16);
    v9 = *(__int64 (**)())(**(_QWORD **)(*v8 + 16LL) + 112LL);
    if ( v9 == sub_1D00B10 )
      BUG();
    result = *(_QWORD *)(v9() + 232);
    if ( *(_BYTE *)(result + 8LL * a2 + 4) )
    {
      result = *(_QWORD *)(v8[38] + 8LL * (a2 >> 6)) & (1LL << a2);
      if ( !result )
      {
        v10 = *(_QWORD *)(a1 + 8);
        v11 = *(_DWORD *)(*(_QWORD *)(v10 + 8) + 24LL * a2 + 16);
        v12 = a2 * (v11 & 0xF);
        result = *(_QWORD *)(v10 + 56) + 2LL * (v11 >> 4);
        v13 = (unsigned __int16 *)(result + 2);
        v14 = *(_WORD *)result + v12;
        while ( 1 )
        {
          v15 = v13;
          if ( !v13 )
            break;
          while ( 1 )
          {
            v16 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
            v17 = sub_1EE52A0(*(_DWORD **)a4, v16, v14);
            if ( (_DWORD *)v16 == v17 )
            {
              if ( v18 >= *(_DWORD *)(a4 + 12) )
              {
                sub_16CD150(a4, (const void *)(a4 + 16), 0, 8, v18, v19);
                v17 = (_DWORD *)(*(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8));
              }
              *(_QWORD *)v17 = v14 | 0xFFFFFFFF00000000LL;
              ++*(_DWORD *)(a4 + 8);
            }
            else
            {
              v17[1] = -1;
            }
            result = *v15;
            v13 = 0;
            ++v15;
            if ( !(_WORD)result )
              break;
            v14 += result;
            if ( !v15 )
              return result;
          }
        }
      }
    }
  }
  else
  {
    if ( a3 )
      v6 = *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 248LL) + 4LL * a3);
    else
      v6 = (unsigned int)sub_1E69F40(*(_QWORD *)(a1 + 16), a2);
    return (__int64)sub_1EE58A0(a4, v5 | (v6 << 32));
  }
  return result;
}

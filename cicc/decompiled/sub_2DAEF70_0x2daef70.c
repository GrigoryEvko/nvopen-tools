// Function: sub_2DAEF70
// Address: 0x2daef70
//
__int64 __fastcall sub_2DAEF70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v7; // r14d
  __int64 v8; // r13
  __int64 v9; // r12
  unsigned int v10; // r15d
  _QWORD *v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r12
  __int64 v15; // rdx
  __int64 *v16; // rcx
  unsigned int *v17; // rax
  __int64 v18; // r12
  unsigned int *v19; // rax
  __int64 *v20; // rdx
  __int64 v21; // rdx

  result = *(unsigned __int8 *)(a2 + 4);
  if ( (result & 1) == 0 && (result & 2) == 0 && ((*(_BYTE *)(a2 + 3) & 0x10) == 0 || (*(_DWORD *)a2 & 0xFFF00) != 0) )
  {
    v7 = *(_DWORD *)(a2 + 8);
    if ( v7 < 0 )
    {
      v8 = a3;
      v9 = a4;
      if ( ((*(_DWORD *)a2 >> 8) & 0xFFF) != 0 )
      {
        v8 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 312LL))(*(_QWORD *)(a1 + 8));
        v9 = a3;
      }
      v10 = v7 & 0x7FFFFFFF;
      result = sub_2EBF1E0(*(_QWORD *)a1, (unsigned int)v7, a3, a4, a5, a6);
      v11 = (_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL * (v7 & 0x7FFFFFFF));
      v13 = v9 & v12;
      v14 = v11[1];
      if ( __PAIR128__(v13 & (unsigned __int64)~v14, result & (unsigned __int64)v8 & ~*v11) != 0 )
      {
        *v11 |= result & v8;
        v15 = 1LL << v7;
        v11[1] = v13 | v14;
        result = v10 >> 6;
        if ( (*(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * result) & (1LL << v7)) != 0 )
        {
          v16 = (__int64 *)(*(_QWORD *)(a1 + 104) + 8 * result);
          result = *v16;
          if ( (*v16 & v15) == 0 )
          {
            *v16 = result | v15;
            v17 = *(unsigned int **)(a1 + 72);
            if ( v17 == (unsigned int *)(*(_QWORD *)(a1 + 88) - 4LL) )
            {
              v18 = *(_QWORD *)(a1 + 96);
              if ( ((__int64)(*(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 40)) >> 2)
                 + ((((v18 - *(_QWORD *)(a1 + 64)) >> 3) - 1) << 7)
                 + (((__int64)v17 - *(_QWORD *)(a1 + 80)) >> 2) == 0x1FFFFFFFFFFFFFFFLL )
                sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
              if ( (unsigned __int64)(*(_QWORD *)(a1 + 32) - ((v18 - *(_QWORD *)(a1 + 24)) >> 3)) <= 1 )
              {
                sub_1D7F850((__int64 *)(a1 + 24), 1u, 0);
                v18 = *(_QWORD *)(a1 + 96);
              }
              *(_QWORD *)(v18 + 8) = sub_22077B0(0x200u);
              v19 = *(unsigned int **)(a1 + 72);
              if ( v19 )
                *v19 = v10;
              v20 = (__int64 *)(*(_QWORD *)(a1 + 96) + 8LL);
              *(_QWORD *)(a1 + 96) = v20;
              result = *v20;
              v21 = *v20 + 512;
              *(_QWORD *)(a1 + 80) = result;
              *(_QWORD *)(a1 + 88) = v21;
              *(_QWORD *)(a1 + 72) = result;
            }
            else
            {
              if ( v17 )
              {
                *v17 = v10;
                v17 = *(unsigned int **)(a1 + 72);
              }
              result = (__int64)(v17 + 1);
              *(_QWORD *)(a1 + 72) = result;
            }
          }
        }
      }
    }
  }
  return result;
}

// Function: sub_2DAE870
// Address: 0x2dae870
//
__int64 __fastcall sub_2DAE870(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rbx
  int v8; // ecx
  unsigned int v9; // r14d
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // r10
  int v13; // r11d
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rdx
  unsigned int *v21; // rax
  __int64 v22; // rbx
  unsigned int *v23; // rax
  __int64 *v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // [rsp-50h] [rbp-50h]
  __int64 v27; // [rsp-48h] [rbp-48h]
  int v29; // [rsp-40h] [rbp-40h]

  result = *(unsigned __int8 *)(a2 + 4);
  if ( (result & 1) == 0 && (result & 2) == 0 && ((*(_BYTE *)(a2 + 3) & 0x10) == 0 || (*(_DWORD *)a2 & 0xFFF00) != 0) )
  {
    v5 = *(_QWORD *)(a2 + 16);
    result = *(_QWORD *)(v5 + 16);
    if ( *(_BYTE *)(result + 4) == 1 && *(_WORD *)(v5 + 68) != 28 )
    {
      result = sub_2E88FE0(v5);
      v8 = *(_DWORD *)(*(_QWORD *)(v5 + 32) + 8LL);
      v27 = *(_QWORD *)(v5 + 32);
      if ( v8 < 0 )
      {
        v9 = v8 & 0x7FFFFFFF;
        v10 = 1LL << v8;
        result = (v8 & 0x7FFFFFFFu) >> 6;
        v26 = 8 * result;
        if ( (*(_QWORD *)(*(_QWORD *)(a1 + 176) + 8 * result) & (1LL << v8)) != 0 )
        {
          v11 = sub_2EAB0A0(a2);
          v12 = v27;
          v13 = v11;
          v14 = (*(_DWORD *)a2 >> 8) & 0xFFF;
          if ( ((*(_DWORD *)a2 >> 8) & 0xFFF) != 0 )
          {
            v29 = v11;
            v18 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a1 + 8) + 320LL))(
                    *(_QWORD *)(a1 + 8),
                    v14,
                    a3,
                    a4);
            v12 = v27;
            v13 = v29;
            a3 = v18;
            a4 = v19;
          }
          v16 = sub_2DAE4E0((_QWORD *)a1, v12, v13, a3, a4);
          result = *(_QWORD *)(a1 + 16) + 32LL * v9;
          v17 = *(_QWORD *)(result + 24);
          if ( v15 & ~v17 | v16 & ~*(_QWORD *)(result + 16) )
          {
            *(_QWORD *)(result + 16) |= v16;
            *(_QWORD *)(result + 24) = v17 | v15;
            v20 = (__int64 *)(*(_QWORD *)(a1 + 104) + v26);
            result = *v20;
            if ( (*v20 & v10) == 0 )
            {
              *v20 = result | v10;
              v21 = *(unsigned int **)(a1 + 72);
              if ( v21 == (unsigned int *)(*(_QWORD *)(a1 + 88) - 4LL) )
              {
                v22 = *(_QWORD *)(a1 + 96);
                if ( ((__int64)(*(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 40)) >> 2)
                   + ((((v22 - *(_QWORD *)(a1 + 64)) >> 3) - 1) << 7)
                   + (((__int64)v21 - *(_QWORD *)(a1 + 80)) >> 2) == 0x1FFFFFFFFFFFFFFFLL )
                  sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
                if ( (unsigned __int64)(*(_QWORD *)(a1 + 32) - ((v22 - *(_QWORD *)(a1 + 24)) >> 3)) <= 1 )
                {
                  sub_1D7F850((__int64 *)(a1 + 24), 1u, 0);
                  v22 = *(_QWORD *)(a1 + 96);
                }
                *(_QWORD *)(v22 + 8) = sub_22077B0(0x200u);
                v23 = *(unsigned int **)(a1 + 72);
                if ( v23 )
                  *v23 = v9;
                v24 = (__int64 *)(*(_QWORD *)(a1 + 96) + 8LL);
                *(_QWORD *)(a1 + 96) = v24;
                result = *v24;
                v25 = *v24 + 512;
                *(_QWORD *)(a1 + 80) = result;
                *(_QWORD *)(a1 + 88) = v25;
                *(_QWORD *)(a1 + 72) = result;
              }
              else
              {
                if ( v21 )
                {
                  *v21 = v9;
                  v21 = *(unsigned int **)(a1 + 72);
                }
                result = (__int64)(v21 + 1);
                *(_QWORD *)(a1 + 72) = result;
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// Function: sub_1A1E350
// Address: 0x1a1e350
//
char __fastcall sub_1A1E350(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v7; // r13
  unsigned __int64 v8; // r8
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 *v11; // rdx
  _DWORD *v12; // rdi
  _DWORD *v13; // rsi
  _DWORD *v14; // rdi
  _DWORD *v15; // rsi
  int v16[11]; // [rsp-2Ch] [rbp-2Ch] BYREF

  if ( a2 == a3 )
    return 1;
  v4 = a3;
  v5 = a2;
  if ( *(_BYTE *)(a2 + 8) != 11 || *(_BYTE *)(a3 + 8) != 11 )
  {
    v7 = sub_127FA20(a1, a3);
    if ( v7 == sub_127FA20(a1, a2) )
    {
      v8 = *(unsigned __int8 *)(v4 + 8);
      if ( (unsigned __int8)v8 <= 0x10u )
      {
        v9 = 100990;
        if ( _bittest64(&v9, v8) )
        {
          v10 = *(unsigned __int8 *)(a2 + 8);
          if ( (unsigned __int8)v10 <= 0x10u )
          {
            if ( _bittest64(&v9, v10) )
            {
              if ( (_BYTE)v10 == 16 )
              {
                v5 = **(_QWORD **)(a2 + 16);
                LOBYTE(v10) = *(_BYTE *)(v5 + 8);
              }
              if ( (_BYTE)v8 == 16 )
              {
                v11 = *(__int64 **)(v4 + 16);
                v4 = *v11;
                LOBYTE(v8) = *(_BYTE *)(*v11 + 8);
              }
              if ( (_BYTE)v8 == 15 )
              {
                if ( (_BYTE)v10 == 15 )
                  return *(_DWORD *)(v5 + 8) >> 8 == *(_DWORD *)(v4 + 8) >> 8;
                if ( (_BYTE)v10 == 11 )
                {
                  v14 = *(_DWORD **)(a1 + 408);
                  v15 = &v14[*(unsigned int *)(a1 + 416)];
                  v16[0] = *(_DWORD *)(v4 + 8) >> 8;
                  return v15 == sub_1A1A6C0(v14, (__int64)v15, v16);
                }
                return (_BYTE)v8 == 11;
              }
              if ( (_BYTE)v10 != 15 )
                return 1;
              v12 = *(_DWORD **)(a1 + 408);
              v13 = &v12[*(unsigned int *)(a1 + 416)];
              v16[0] = *(_DWORD *)(v5 + 8) >> 8;
              if ( v13 == sub_1A1A6C0(v12, (__int64)v13, v16) )
                return (_BYTE)v8 == 11;
            }
          }
        }
      }
    }
    return 0;
  }
  return *(_DWORD *)(a3 + 8) >> 8 <= 8u && *(_DWORD *)(a3 + 8) >> 8 >= *(_DWORD *)(a2 + 8) >> 8;
}

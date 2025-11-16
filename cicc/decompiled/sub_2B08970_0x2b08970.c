// Function: sub_2B08970
// Address: 0x2b08970
//
bool __fastcall sub_2B08970(__int64 a1, __int64 a2, __int64 a3)
{
  bool result; // al
  unsigned __int8 *v5; // r15
  __int64 v6; // r13
  __int64 v7; // rdi
  __int64 v8; // r12
  unsigned __int8 v9; // cl
  __int64 v10; // rcx
  unsigned __int8 v11; // si
  unsigned int v12; // r14d
  unsigned int v13; // r8d
  int v14; // eax
  unsigned __int8 v15; // dl
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rsi
  unsigned int v19; // ecx
  unsigned int v20; // r9d
  __int64 v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // r10
  unsigned int v24; // ecx
  __int64 v25; // rsi

  result = 1;
  v5 = *(unsigned __int8 **)(a2 - 64);
  v6 = *(_QWORD *)(a3 - 64);
  v7 = *((_QWORD *)v5 + 1);
  v8 = *(_QWORD *)(v6 + 8);
  v9 = *(_BYTE *)(v8 + 8);
  if ( *(_BYTE *)(v7 + 8) >= v9 )
  {
    result = 0;
    if ( *(_BYTE *)(v7 + 8) <= v9 )
    {
      v10 = *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL);
      result = 1;
      v11 = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a3 - 32) + 8LL) + 8LL);
      if ( *(_BYTE *)(v10 + 8) >= v11 )
      {
        result = 0;
        if ( *(_BYTE *)(v10 + 8) <= v11 )
        {
          v12 = sub_BCB060(v7);
          v13 = sub_BCB060(v8);
          result = 1;
          if ( v12 >= v13 )
          {
            result = 0;
            if ( v12 <= v13 )
            {
              v14 = *v5;
              v15 = *(_BYTE *)v6;
              if ( (unsigned __int8)v14 <= 0x1Cu || v15 <= 0x1Cu )
              {
                return (unsigned __int8)v14 < v15;
              }
              else
              {
                v16 = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
                v17 = *((_QWORD *)v5 + 5);
                if ( v17 )
                {
                  v18 = (unsigned int)(*(_DWORD *)(v17 + 44) + 1);
                  v19 = *(_DWORD *)(v17 + 44) + 1;
                }
                else
                {
                  v18 = 0;
                  v19 = 0;
                }
                v20 = *(_DWORD *)(v16 + 32);
                v21 = 0;
                if ( v19 < v20 )
                  v21 = *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8 * v18);
                v22 = *(_QWORD *)(v6 + 40);
                if ( v22 )
                {
                  v23 = (unsigned int)(*(_DWORD *)(v22 + 44) + 1);
                  v24 = *(_DWORD *)(v22 + 44) + 1;
                }
                else
                {
                  v23 = 0;
                  v24 = 0;
                }
                v25 = 0;
                if ( v20 > v24 )
                  v25 = *(_QWORD *)(*(_QWORD *)(v16 + 24) + 8 * v23);
                if ( v25 == v21 )
                  return v14 - 29 < (unsigned int)v15 - 29;
                else
                  return *(_DWORD *)(v21 + 72) < *(_DWORD *)(v25 + 72);
              }
            }
          }
        }
      }
    }
  }
  return result;
}

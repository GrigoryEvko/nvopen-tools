// Function: sub_28C8B00
// Address: 0x28c8b00
//
__int64 *__fastcall sub_28C8B00(__int64 a1, __int64 *a2)
{
  int v2; // ebx
  __int64 v3; // r14
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rbx
  __int64 *v9; // r12
  __int64 v10; // r13
  int i; // r9d
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v16; // eax
  int v17; // eax
  char v18; // al
  int v19; // [rsp+Ch] [rbp-44h]
  int v20; // [rsp+Ch] [rbp-44h]
  int v21; // [rsp+Ch] [rbp-44h]
  __int64 v22; // [rsp+10h] [rbp-40h]
  __int64 v23; // [rsp+10h] [rbp-40h]
  __int64 v24; // [rsp+10h] [rbp-40h]
  unsigned int v25; // [rsp+18h] [rbp-38h]
  unsigned int v26; // [rsp+18h] [rbp-38h]
  __int64 v27; // [rsp+18h] [rbp-38h]
  unsigned int v28; // [rsp+18h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 24);
  if ( !v2 )
    return 0;
  v3 = *a2;
  v4 = *(_QWORD *)(a1 + 8);
  LODWORD(v6) = *(_QWORD *)(*a2 + 16);
  if ( !(_DWORD)v6 )
  {
    v27 = *(_QWORD *)(a1 + 8);
    v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 32LL))(v3);
    v4 = v27;
    *(_QWORD *)(v3 + 16) = v6;
    v3 = *a2;
  }
  v7 = (unsigned int)(v2 - 1);
  v8 = (unsigned int)v7 & (unsigned int)v6;
  v9 = (__int64 *)(v4 + 16 * v8);
  v10 = *v9;
  if ( v3 != *v9 )
  {
    for ( i = 1; ; ++i )
    {
      if ( v3 != -8 && v10 != 0x7FFFFFFF0LL && v3 != 0x7FFFFFFF0LL && v10 != -8 )
      {
        v12 = *(_QWORD *)(v10 + 16);
        if ( !(_DWORD)v12 )
        {
          v19 = i;
          v22 = v4;
          v25 = v7;
          v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v10 + 32LL))(
                  v10,
                  v12,
                  v4,
                  v7,
                  0x7FFFFFFF0LL);
          i = v19;
          v4 = v22;
          *(_QWORD *)(v10 + 16) = v13;
          v7 = v25;
          v12 = v13;
        }
        v14 = *(_QWORD *)(v3 + 16);
        if ( !(_DWORD)v14 )
        {
          v20 = i;
          v23 = v4;
          v26 = v7;
          v14 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v3 + 32LL))(
                  v3,
                  v12,
                  v4,
                  v7,
                  0x7FFFFFFF0LL);
          i = v20;
          v4 = v23;
          v7 = v26;
          *(_QWORD *)(v3 + 16) = v14;
        }
        if ( v14 == v12 )
        {
          v16 = *(_DWORD *)(v3 + 12);
          if ( v16 == *(_DWORD *)(v10 + 12) )
          {
            if ( v16 > 0xFFFFFFFD )
              break;
            v17 = *(_DWORD *)(v3 + 8);
            if ( (unsigned int)(v17 - 11) <= 1 || v17 == *(_DWORD *)(v10 + 8) )
            {
              v21 = i;
              v24 = v4;
              v28 = v7;
              v18 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v3 + 16LL))(
                      v3,
                      v10,
                      v4,
                      v7,
                      0x7FFFFFFF0LL);
              v7 = v28;
              v4 = v24;
              i = v21;
              if ( v18 )
                break;
            }
          }
        }
      }
      if ( *v9 == -8 )
        return 0;
      v3 = *a2;
      v8 = (unsigned int)v7 & ((_DWORD)v8 + i);
      v9 = (__int64 *)(v4 + 16 * v8);
      v10 = *v9;
      if ( *v9 == *a2 )
        break;
    }
  }
  return v9;
}

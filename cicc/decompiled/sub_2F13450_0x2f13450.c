// Function: sub_2F13450
// Address: 0x2f13450
//
__int64 __fastcall sub_2F13450(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // r9
  __int64 v7; // r13
  __int64 v8; // r15
  __int64 v9; // rbx
  __m128i *v10; // rsi
  __int64 v11; // rbx
  __m128i *v12; // rdx
  __int64 v13; // rax
  const char *v14; // rdi
  __int64 v15; // [rsp+20h] [rbp-E0h]
  __int64 v16; // [rsp+28h] [rbp-D8h]
  __m128i *v18; // [rsp+40h] [rbp-C0h]
  _QWORD v20[4]; // [rsp+50h] [rbp-B0h]
  _QWORD v21[4]; // [rsp+70h] [rbp-90h]
  _QWORD v22[14]; // [rsp+90h] [rbp-70h] BYREF

  v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a3 + 16) + 200LL))(*(_QWORD *)(a3 + 16));
  result = *(_QWORD *)(a3 + 752);
  v6 = 32LL * *(unsigned int *)(a3 + 760);
  v7 = result + v6;
  if ( result != result + v6 )
  {
    while ( *(_BYTE *)(result + 4) != 1 )
    {
      result += 32;
      if ( v7 == result )
        return result;
    }
    if ( v7 != result )
    {
      v8 = result;
      v9 = v7;
      do
      {
        v10 = (__m128i *)a2[51];
        if ( v10 == (__m128i *)a2[52] )
        {
          sub_2F12EA0(a2 + 50, v10);
          v18 = (__m128i *)a2[51];
        }
        else
        {
          if ( v10 )
          {
            memset(v10, 0, 0xC0u);
            v10->m128i_i64[0] = (__int64)v10[1].m128i_i64;
            v10[3].m128i_i64[0] = (__int64)v10[4].m128i_i64;
            v10[6].m128i_i64[0] = (__int64)v10[7].m128i_i64;
            v10[9].m128i_i64[0] = (__int64)v10[10].m128i_i64;
            v10 = (__m128i *)a2[51];
          }
          v18 = v10 + 12;
          a2[51] = (unsigned __int64)&v10[12];
        }
        v16 = v9;
        v11 = 0;
        v12 = v18 - 9;
        v20[1] = v18 - 6;
        v20[2] = v18 - 3;
        v13 = *(_QWORD *)(v8 + 8);
        v20[0] = v18 - 9;
        v21[0] = v13;
        v21[1] = *(_QWORD *)(v8 + 16);
        v21[2] = *(_QWORD *)(v8 + 24);
        while ( 1 )
        {
          v22[6] = v12;
          memset(&v22[1], 0, 32);
          v22[5] = 0x100000000LL;
          v22[0] = &unk_49DD210;
          sub_CB5980((__int64)v22, 0, 0, 0);
          v14 = (const char *)v21[v11++];
          sub_A61DC0(v14, (__int64)v22, a4, 0);
          v22[0] = &unk_49DD210;
          sub_CB5840((__int64)v22);
          if ( v11 == 3 )
            break;
          v12 = (__m128i *)v20[v11];
        }
        v9 = v16;
        if ( *(_BYTE *)(v8 + 4) != 1 )
          abort();
        sub_2F07630(*(_DWORD *)v8, (__int64)v18[-12].m128i_i64, v15);
        result = v8 + 32;
        if ( v16 == v8 + 32 )
          break;
        while ( 1 )
        {
          v8 = result;
          if ( *(_BYTE *)(result + 4) == 1 )
            break;
          result += 32;
          if ( v16 == result )
            return result;
        }
      }
      while ( result != v16 );
    }
  }
  return result;
}

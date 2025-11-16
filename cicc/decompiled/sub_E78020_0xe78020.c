// Function: sub_E78020
// Address: 0xe78020
//
unsigned __int64 __fastcall sub_E78020(
        __int64 a1,
        unsigned __int64 a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rsi
  unsigned __int64 v8; // rcx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r8
  char v13; // r13
  unsigned __int32 v14; // ebx
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int16 v18; // bx
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  int v25; // [rsp-30h] [rbp-30h]
  int v26; // [rsp-30h] [rbp-30h]
  __int16 v27; // [rsp-30h] [rbp-30h]
  unsigned __int64 v28; // [rsp-30h] [rbp-30h]
  unsigned __int64 v29; // [rsp-30h] [rbp-30h]

  result = a2;
  v7 = *(_QWORD *)(a1 + 152);
  v8 = *(unsigned int *)(v7 + 28);
  if ( result >= v8 )
  {
    result /= v8;
    v10 = a3[1];
    v11 = a3[2];
    v12 = v10 + 1;
    if ( result <= 0x3F )
    {
      if ( v12 > v11 )
      {
        v26 = result;
        sub_C8D290((__int64)a3, a3 + 3, v10 + 1, 1u, v12, a6);
        v10 = a3[1];
        LODWORD(result) = v26;
      }
      result = (unsigned int)result | 0x40;
      *(_BYTE *)(*a3 + v10) = result;
      ++a3[1];
    }
    else if ( (result & 0xFFFFFFFFFFFFFF00LL) != 0 )
    {
      v13 = *(_BYTE *)(v7 + 16);
      if ( (result & 0xFFFFFFFFFFFF0000LL) != 0 )
      {
        if ( v12 > v11 )
        {
          v25 = result;
          sub_C8D290((__int64)a3, a3 + 3, v10 + 1, 1u, v12, a6);
          v10 = a3[1];
          LODWORD(result) = v25;
        }
        v14 = _byteswap_ulong(result);
        if ( v13 )
          v14 = result;
        *(_BYTE *)(*a3 + v10) = 4;
        v15 = a3[1];
        v16 = v15 + 1;
        v17 = v15 + 5;
        a3[1] = v16;
        if ( v17 > a3[2] )
        {
          sub_C8D290((__int64)a3, a3 + 3, v17, 1u, v12, a6);
          v16 = a3[1];
        }
        result = *a3;
        *(_DWORD *)(*a3 + v16) = v14;
        a3[1] += 4LL;
      }
      else
      {
        if ( v12 > v11 )
        {
          v27 = result;
          sub_C8D290((__int64)a3, a3 + 3, v10 + 1, 1u, v12, a6);
          v10 = a3[1];
          LOWORD(result) = v27;
        }
        v18 = __ROL2__(result, 8);
        *(_BYTE *)(*a3 + v10) = 3;
        v19 = a3[1];
        if ( v13 )
          v18 = result;
        v20 = v19 + 1;
        v21 = v19 + 3;
        a3[1] = v20;
        if ( v21 > a3[2] )
        {
          sub_C8D290((__int64)a3, a3 + 3, v21, 1u, v12, a6);
          v20 = a3[1];
        }
        result = *a3;
        *(_WORD *)(*a3 + v20) = v18;
        a3[1] += 2LL;
      }
    }
    else
    {
      if ( v12 > v11 )
      {
        v29 = result;
        sub_C8D290((__int64)a3, a3 + 3, v10 + 1, 1u, v12, a6);
        v10 = a3[1];
        result = v29;
      }
      *(_BYTE *)(*a3 + v10) = 2;
      v22 = a3[1];
      v23 = v22 + 1;
      v24 = v22 + 2;
      a3[1] = v23;
      if ( v24 > a3[2] )
      {
        v28 = result;
        sub_C8D290((__int64)a3, a3 + 3, v24, 1u, v12, a6);
        v23 = a3[1];
        result = v28;
      }
      *(_BYTE *)(*a3 + v23) = result;
      ++a3[1];
    }
  }
  return result;
}

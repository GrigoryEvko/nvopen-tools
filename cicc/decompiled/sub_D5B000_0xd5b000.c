// Function: sub_D5B000
// Address: 0xd5b000
//
__int64 *__fastcall sub_D5B000(__int64 *a1, _QWORD *a2)
{
  __int64 v4; // rsi
  __int64 *result; // rax
  __int64 *v6; // rcx
  __int64 *v7; // r8
  __int64 *v8; // rdi
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // r13
  __int64 **v12; // rdx
  __int64 v13; // rdx
  _QWORD *v14; // [rsp+8h] [rbp-48h] BYREF
  __int64 *v15[8]; // [rsp+10h] [rbp-40h] BYREF

  v4 = *a2;
  result = (__int64 *)a1[73];
  if ( v4 )
  {
    v6 = (__int64 *)a1[75];
    v7 = (__int64 *)a1[76];
    v8 = (__int64 *)a1[77];
LABEL_3:
    if ( v8 != result )
    {
      while ( 1 )
      {
        v9 = *result++;
        if ( v9 == v4 )
          break;
        if ( result != v6 )
          goto LABEL_3;
        result = (__int64 *)v7[1];
        ++v7;
        v6 = result + 64;
        if ( v8 == result )
          return result;
      }
      if ( result == v6 )
      {
        result = (__int64 *)v7[1];
        ++v7;
      }
      v14 = a2;
      v15[0] = result;
      v10 = (__int64 *)*v7;
      v15[3] = v7;
      v15[1] = v10;
      v15[2] = v10 + 64;
      return sub_D59F80(a1 + 71, v15, 1u, (__int64 *)&v14);
    }
  }
  else if ( (__int64 *)a1[74] == result )
  {
    v11 = a1[76];
    if ( ((a1[77] - a1[78]) >> 3) + ((((a1[80] - v11) >> 3) - 1) << 6) + ((a1[75] - (__int64)result) >> 3) == 0xFFFFFFFFFFFFFFFLL )
      sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
    if ( v11 == a1[71] )
    {
      sub_D58C10(a1 + 71, 1u, 1);
      v11 = a1[76];
    }
    *(_QWORD *)(v11 - 8) = sub_22077B0(512);
    v12 = (__int64 **)(a1[76] - 8);
    a1[76] = (__int64)v12;
    result = *v12;
    v13 = (__int64)(*v12 + 64);
    a1[74] = (__int64)result;
    a1[75] = v13;
    a1[73] = (__int64)(result + 63);
    result[63] = (__int64)a2;
  }
  else
  {
    *(result - 1) = (__int64)a2;
    a1[73] -= 8;
  }
  return result;
}

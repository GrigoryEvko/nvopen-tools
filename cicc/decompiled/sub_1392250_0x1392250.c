// Function: sub_1392250
// Address: 0x1392250
//
__int64 __fastcall sub_1392250(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 **v4; // rax
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 **v7; // rax
  __int64 *v8; // r15
  __int64 result; // rax
  unsigned __int8 v10; // al
  __int64 v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 *v18; // [rsp+0h] [rbp-40h] BYREF
  unsigned int v19; // [rsp+8h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 8);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v4 = *(__int64 ***)(a2 - 8);
  else
    v4 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v5 = **v4;
  if ( *(_BYTE *)(v5 + 8) == 16 )
    v5 = **(_QWORD **)(v5 + 16);
  v19 = 8 * sub_15A9520(v3, *(_DWORD *)(v5 + 8) >> 8);
  if ( v19 > 0x40 )
    sub_16A4EF0(&v18, 0, 0);
  else
    v18 = 0;
  v6 = 0x7FFFFFFFFFFFFFFFLL;
  if ( (unsigned __int8)sub_1634900(a2, *(_QWORD *)(a1 + 8), &v18) )
  {
    if ( v19 > 0x40 )
      v6 = *v18;
    else
      v6 = (__int64)((_QWORD)v18 << (64 - (unsigned __int8)v19)) >> (64 - (unsigned __int8)v19);
  }
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v7 = *(__int64 ***)(a2 - 8);
  else
    v7 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v8 = *v7;
  result = **v7;
  if ( *(_BYTE *)(result + 8) == 15 )
  {
    result = *(_QWORD *)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
    {
      v10 = *((_BYTE *)v8 + 16);
      if ( v10 > 3u )
      {
        if ( v10 == 5 )
        {
          result = (unsigned int)*((unsigned __int16 *)v8 + 9) - 51;
          if ( (unsigned int)result > 1 )
          {
            result = sub_13848E0(*(_QWORD *)(a1 + 24), (__int64)v8, 0, 0);
            if ( (_BYTE)result )
              result = sub_1391610(a1, (__int64)v8, v15);
          }
        }
        else
        {
          result = sub_13848E0(*(_QWORD *)(a1 + 24), (__int64)v8, 0, 0);
        }
      }
      else
      {
        v11 = *(_QWORD *)(a1 + 24);
        v12 = sub_14C81A0(v8);
        v13 = v11;
        result = sub_13848E0(v11, (__int64)v8, 0, v12);
        if ( (_BYTE)result )
        {
          v16 = *(_QWORD *)(a1 + 24);
          v17 = sub_14C8160(v13, v8, v14);
          result = sub_13848E0(v16, (__int64)v8, 1u, v17);
        }
      }
      if ( v8 != (__int64 *)a2 )
        result = (__int64)sub_1391C50(a1, (__int64)v8, a2, (__m128i *)v6);
    }
  }
  if ( v19 > 0x40 )
  {
    if ( v18 )
      return j_j___libc_free_0_0(v18);
  }
  return result;
}

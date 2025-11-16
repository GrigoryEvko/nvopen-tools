// Function: sub_164EF10
// Address: 0x164ef10
//
unsigned __int64 __fastcall sub_164EF10(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  unsigned __int8 *v3; // r13
  __int64 v6; // r14
  _BYTE *v7; // rax
  __int64 v8; // rsi
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  const char *v12; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]
  char v14; // [rsp+11h] [rbp-2Fh]

  result = 1LL - *(unsigned int *)(a2 + 8);
  v3 = *(unsigned __int8 **)(a2 + 8 * result);
  if ( v3 )
  {
    result = *v3;
    if ( (unsigned __int8)result > 0xEu )
    {
      result = (unsigned int)(result - 32);
      if ( (unsigned __int8)result <= 1u )
        return result;
    }
    else if ( (unsigned __int8)result > 0xAu )
    {
      return result;
    }
    v6 = *(_QWORD *)a1;
    v14 = 1;
    v12 = "invalid type ref";
    v13 = 3;
    if ( v6 )
    {
      sub_16E2CE0(&v12, v6);
      v7 = *(_BYTE **)(v6 + 24);
      if ( (unsigned __int64)v7 >= *(_QWORD *)(v6 + 16) )
      {
        sub_16E7DE0(v6, 10);
      }
      else
      {
        *(_QWORD *)(v6 + 24) = v7 + 1;
        *v7 = 10;
      }
      v8 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 73) = 1;
      *(_BYTE *)(a1 + 72) |= result;
      if ( v8 )
      {
        sub_15562E0((unsigned __int8 *)a2, v8, a1 + 16, *(_QWORD *)(a1 + 8));
        v9 = *(_QWORD *)a1;
        v10 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          sub_16E7DE0(v9, 10);
        }
        else
        {
          *(_QWORD *)(v9 + 24) = v10 + 1;
          *v10 = 10;
        }
        sub_15562E0(v3, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
        v11 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
        {
          return sub_16E7DE0(v11, 10);
        }
        else
        {
          *(_QWORD *)(v11 + 24) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
    }
    else
    {
      *(_BYTE *)(a1 + 73) = 1;
      result = *(unsigned __int8 *)(a1 + 74);
      *(_BYTE *)(a1 + 72) |= result;
    }
  }
  return result;
}

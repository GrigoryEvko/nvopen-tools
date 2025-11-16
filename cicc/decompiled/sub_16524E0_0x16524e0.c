// Function: sub_16524E0
// Address: 0x16524e0
//
unsigned __int64 __fastcall sub_16524E0(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  __int64 v6; // rdx
  unsigned __int64 result; // rax
  unsigned __int8 *v8; // r14
  __int64 v9; // rbx
  _BYTE *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  _BYTE *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  _QWORD v16[2]; // [rsp+0h] [rbp-50h] BYREF
  char v17; // [rsp+10h] [rbp-40h]
  char v18; // [rsp+11h] [rbp-3Fh]

  if ( *(_BYTE *)a3 == 4 )
  {
    v6 = 8LL * *(unsigned int *)(a3 + 8);
    result = a3 - v6;
    if ( a3 != a3 - v6 )
    {
      while ( 1 )
      {
        v8 = *(unsigned __int8 **)result;
        if ( !*(_QWORD *)result || (unsigned int)*v8 - 22 > 1 )
          break;
        result += 8LL;
        if ( a3 == result )
          return result;
      }
      v9 = *(_QWORD *)a1;
      v18 = 1;
      v16[0] = "invalid template parameter";
      v17 = 3;
      if ( v9 )
      {
        sub_16E2CE0(v16, v9);
        v10 = *(_BYTE **)(v9 + 24);
        if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 16) )
        {
          sub_16E7DE0(v9, 10);
        }
        else
        {
          *(_QWORD *)(v9 + 24) = v10 + 1;
          *v10 = 10;
        }
        v11 = *(_QWORD *)a1;
        result = *(unsigned __int8 *)(a1 + 74);
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= result;
        if ( v11 )
        {
          sub_15562E0(a2, v11, a1 + 16, *(_QWORD *)(a1 + 8));
          v12 = *(_QWORD *)a1;
          v13 = *(_BYTE **)(*(_QWORD *)a1 + 24LL);
          if ( (unsigned __int64)v13 >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
            sub_16E7DE0(v12, 10);
          }
          else
          {
            *(_QWORD *)(v12 + 24) = v13 + 1;
            *v13 = 10;
          }
          sub_15562E0((unsigned __int8 *)a3, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
          v14 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
          {
            result = sub_16E7DE0(v14, 10);
          }
          else
          {
            *(_QWORD *)(v14 + 24) = result + 1;
            *(_BYTE *)result = 10;
          }
          if ( v8 )
          {
            sub_15562E0(v8, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
            v15 = *(_QWORD *)a1;
            result = *(_QWORD *)(*(_QWORD *)a1 + 24LL);
            if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 16LL) )
            {
              return sub_16E7DE0(v15, 10);
            }
            else
            {
              *(_QWORD *)(v15 + 24) = result + 1;
              *(_BYTE *)result = 10;
            }
          }
        }
      }
      else
      {
        result = *(unsigned __int8 *)(a1 + 74);
        *(_BYTE *)(a1 + 73) = 1;
        *(_BYTE *)(a1 + 72) |= result;
      }
    }
  }
  else
  {
    v18 = 1;
    v16[0] = "invalid template params";
    v17 = 3;
    result = sub_16521E0((__int64 *)a1, (__int64)v16);
    if ( *(_QWORD *)a1 )
    {
      sub_164ED40((__int64 *)a1, a2);
      return (unsigned __int64)sub_164ED40((__int64 *)a1, (unsigned __int8 *)a3);
    }
  }
  return result;
}

// Function: sub_1C061F0
// Address: 0x1c061f0
//
__int64 __fastcall sub_1C061F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r15
  unsigned int v5; // r13d
  char *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  char *v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rax
  int v14; // [rsp+1Ch] [rbp-74h]
  __int64 v15; // [rsp+30h] [rbp-60h]
  const char *v17; // [rsp+40h] [rbp-50h] BYREF
  __int64 v18; // [rsp+48h] [rbp-48h]
  char v19[64]; // [rsp+50h] [rbp-40h] BYREF

  result = sub_157EBA0(a3);
  if ( result )
  {
    v4 = result;
    result = sub_15F4D60(result);
    v14 = result;
    if ( (_DWORD)result )
    {
      v5 = 0;
      do
      {
        while ( 1 )
        {
          sub_223E0D0(a2, "\"", 1);
          v6 = (char *)sub_1649960(a3);
          v17 = v19;
          if ( v6 )
          {
            sub_1C04B10((__int64 *)&v17, v6, (__int64)&v6[v7]);
            v8 = sub_223E0D0(a2, v17, v18);
          }
          else
          {
            v18 = 0;
            v19[0] = 0;
            v8 = sub_223E0D0(a2, v19, 0);
          }
          sub_223E0D0(v8, "\"", 1);
          if ( v17 != v19 )
            j_j___libc_free_0(v17, *(_QWORD *)v19 + 1LL);
          sub_223E0D0(a2, " -> ", 4);
          sub_223E0D0(a2, "\"", 1);
          v9 = sub_15F4DF0(v4, v5);
          v10 = (char *)sub_1649960(v9);
          v17 = v19;
          if ( v10 )
          {
            sub_1C04B10((__int64 *)&v17, v10, (__int64)&v10[v11]);
            v12 = sub_223E0D0(a2, v17, v18);
          }
          else
          {
            v18 = 0;
            v19[0] = 0;
            v12 = sub_223E0D0(a2, v19, 0);
          }
          sub_223E0D0(v12, "\" is ", 5);
          if ( v17 != v19 )
            j_j___libc_free_0(v17, *(_QWORD *)v19 + 1LL);
          v15 = *(_QWORD *)(a1 + 160);
          v13 = sub_15F4DF0(v4, v5);
          if ( !(unsigned __int8)sub_1C05930(*(_QWORD *)(*(_QWORD *)(v15 + 8) + 104LL), a3, v13) )
            break;
          ++v5;
          result = sub_223E0D0(a2, "convergent.\n", 12);
          if ( v5 == v14 )
            return result;
        }
        ++v5;
        result = sub_223E0D0(a2, "not convergent.\n", 16);
      }
      while ( v5 != v14 );
    }
  }
  return result;
}

// Function: sub_3892130
// Address: 0x3892130
//
__int64 __fastcall sub_3892130(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  int v4; // eax
  unsigned __int64 v5; // rax
  int v6; // r8d
  int v7; // r9d
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rax
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // rax
  unsigned __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-50h] BYREF
  char v18; // [rsp+30h] [rbp-40h]
  char v19; // [rsp+31h] [rbp-3Fh]

  v3 = a1 + 8;
  v4 = sub_3887100(a1 + 8);
  *(_DWORD *)(a1 + 64) = v4;
  if ( v4 == 9 )
  {
    *(_DWORD *)(a1 + 64) = sub_3887100(v3);
    return 0;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 56);
    v19 = 1;
    v15 = v5;
    v16 = 0;
    v17[0] = "expected type";
    v18 = 3;
    if ( (unsigned __int8)sub_3891B00(a1, &v16, (__int64)v17, 0) )
      return 1;
    v9 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v9 >= *(_DWORD *)(a2 + 12) )
    {
      sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v6, v7);
      v9 = *(unsigned int *)(a2 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a2 + 8 * v9) = v16;
    v10 = v16;
    ++*(_DWORD *)(a2 + 8);
    if ( sub_1643C40(v10) )
    {
      if ( *(_DWORD *)(a1 + 64) == 4 )
      {
        while ( 1 )
        {
          *(_DWORD *)(a1 + 64) = sub_3887100(v3);
          v11 = *(_QWORD *)(a1 + 56);
          v19 = 1;
          v15 = v11;
          v17[0] = "expected type";
          v18 = 3;
          if ( (unsigned __int8)sub_3891B00(a1, &v16, (__int64)v17, 0) )
            return 1;
          if ( !sub_1643C40(v16) )
            goto LABEL_16;
          v14 = *(unsigned int *)(a2 + 8);
          if ( (unsigned int)v14 >= *(_DWORD *)(a2 + 12) )
          {
            sub_16CD150(a2, (const void *)(a2 + 16), 0, 8, v12, v13);
            v14 = *(unsigned int *)(a2 + 8);
          }
          *(_QWORD *)(*(_QWORD *)a2 + 8 * v14) = v16;
          ++*(_DWORD *)(a2 + 8);
          if ( *(_DWORD *)(a1 + 64) != 4 )
            return sub_388AF10(a1, 9, "expected '}' at end of struct");
        }
      }
      return sub_388AF10(a1, 9, "expected '}' at end of struct");
    }
    else
    {
LABEL_16:
      v19 = 1;
      v18 = 3;
      v17[0] = "invalid element type for struct";
      return sub_38814C0(v3, v15, (__int64)v17);
    }
  }
}

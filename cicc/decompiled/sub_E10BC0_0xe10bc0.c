// Function: sub_E10BC0
// Address: 0xe10bc0
//
__int64 __fastcall sub_E10BC0(__int64 a1)
{
  char *v1; // rdx
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  signed __int64 v5; // r13
  __int64 v6; // rdx
  __int64 v7; // r12
  __int64 result; // rax
  char *v9; // rdx
  char v10; // r8
  char *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  char v14; // cl
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  signed __int64 v19; // r12
  char *v20; // rax
  char v21; // cl
  __int64 v22; // [rsp+8h] [rbp-28h]

  v1 = *(char **)a1;
  if ( *(_QWORD *)(a1 + 8) - *(_QWORD *)a1 > 2u && *(_WORD *)v1 == 28774 && v1[2] == 84 )
  {
    v12 = a1 + 816;
    *(_QWORD *)(v12 - 816) = v1 + 3;
    return sub_E0FD70(v12, "this");
  }
  else if ( (unsigned __int8)sub_E0F5E0((const void **)a1, 2u, "fp") )
  {
    sub_E0E0E0(a1);
    v5 = sub_E0DEF0((char **)a1, 0);
    v7 = v6;
    result = 0;
    v9 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v9 == 95 )
    {
      v13 = (__int64)(v9 + 1);
      *(_QWORD *)a1 = v13;
      result = sub_E0E790(a1 + 816, 32, v13, v2, v3, v4);
      if ( result )
      {
        *(_QWORD *)(result + 16) = v5;
        *(_WORD *)(result + 8) = 16451;
        v14 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 24) = v7;
        *(_QWORD *)result = &unk_49E06E8;
        *(_BYTE *)(result + 10) = v14 & 0xF0 | 5;
      }
    }
  }
  else
  {
    v10 = sub_E0F5E0((const void **)a1, 2u, "fL");
    result = 0;
    if ( v10 )
    {
      if ( sub_E0DEF0((char **)a1, 0)
        && (v11 = *(char **)a1, *(_QWORD *)a1 != *(_QWORD *)(a1 + 8))
        && *v11 == 112
        && (*(_QWORD *)a1 = v11 + 1,
            sub_E0E0E0(a1),
            v19 = sub_E0DEF0((char **)a1, 0),
            v20 = *(char **)a1,
            *(_QWORD *)a1 != *(_QWORD *)(a1 + 8))
        && *v20 == 95 )
      {
        v22 = v15;
        *(_QWORD *)a1 = v20 + 1;
        result = sub_E0E790(a1 + 816, 32, v15, v16, v17, v18);
        if ( result )
        {
          v21 = *(_BYTE *)(result + 10);
          *(_QWORD *)(result + 16) = v19;
          *(_WORD *)(result + 8) = 16451;
          *(_QWORD *)(result + 24) = v22;
          *(_BYTE *)(result + 10) = v21 & 0xF0 | 5;
          *(_QWORD *)result = &unk_49E06E8;
        }
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}

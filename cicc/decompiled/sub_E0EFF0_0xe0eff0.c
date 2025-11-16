// Function: sub_E0EFF0
// Address: 0xe0eff0
//
__int64 __fastcall sub_E0EFF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  signed __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 result; // rax
  __int64 v10; // r14
  char *v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdi
  char v14; // dl

  v6 = sub_E0DEF0((char **)a1, 1);
  result = 0;
  if ( v6 )
  {
    v10 = v4;
    v11 = *(char **)a1;
    if ( *(_QWORD *)a1 != *(_QWORD *)(a1 + 8) && *v11 == 69 )
    {
      v12 = (__int64)(v11 + 1);
      v13 = a1 + 816;
      *(_QWORD *)(v13 - 816) = v12;
      result = sub_E0E790(v13, 48, v12, v5, v7, v8);
      if ( result )
      {
        *(_QWORD *)(result + 16) = a2;
        *(_WORD *)(result + 8) = 16461;
        v14 = *(_BYTE *)(result + 10);
        *(_QWORD *)(result + 24) = a3;
        *(_QWORD *)(result + 32) = v6;
        *(_QWORD *)(result + 40) = v10;
        *(_BYTE *)(result + 10) = v14 & 0xF0 | 5;
        *(_QWORD *)result = &unk_49E0B68;
      }
    }
  }
  return result;
}

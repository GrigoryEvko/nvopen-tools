// Function: sub_824240
// Address: 0x824240
//
__int64 __fastcall sub_824240(const char *a1)
{
  size_t v1; // rbx
  size_t v2; // r13
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  size_t v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 result; // rax

  v1 = strlen(a1);
  v2 = v1;
  sub_823800(qword_4F1F630);
  if ( v1 )
  {
    v6 = 0;
    while ( a1[v6] != 58 )
    {
      if ( v1 == ++v6 )
        goto LABEL_6;
    }
    v2 = v6;
  }
LABEL_6:
  sub_8238B0((_QWORD *)qword_4F1F630, a1, v2, v3, v4, v5);
  v11 = qword_4F1F630;
  v12 = *(_QWORD *)(qword_4F1F630 + 16);
  if ( (unsigned __int64)(v12 + 1) > *(_QWORD *)(qword_4F1F630 + 8) )
  {
    sub_823810((_QWORD *)qword_4F1F630, v12 + 1, v7, v8, v9, v10);
    v11 = qword_4F1F630;
    v12 = *(_QWORD *)(qword_4F1F630 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v11 + 32) + v12) = 0;
  result = *(_QWORD *)(v11 + 32);
  ++*(_QWORD *)(v11 + 16);
  return result;
}

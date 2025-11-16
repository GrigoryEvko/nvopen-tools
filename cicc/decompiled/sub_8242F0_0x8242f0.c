// Function: sub_8242F0
// Address: 0x8242f0
//
__int64 __fastcall sub_8242F0(const char *a1)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  __int64 v4; // r9
  char v5; // al
  size_t v6; // rdx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 result; // rax

  sub_823800(qword_4F1F628);
  do
  {
    v5 = *a1;
    if ( !*a1 )
    {
      v6 = 0;
      goto LABEL_5;
    }
    ++a1;
  }
  while ( v5 != 58 );
  v6 = strlen(a1);
LABEL_5:
  sub_8238B0((_QWORD *)qword_4F1F628, a1, v6, v2, v3, v4);
  v11 = qword_4F1F628;
  v12 = *(_QWORD *)(qword_4F1F628 + 16);
  if ( (unsigned __int64)(v12 + 1) > *(_QWORD *)(qword_4F1F628 + 8) )
  {
    sub_823810((_QWORD *)qword_4F1F628, v12 + 1, v7, v8, v9, v10);
    v11 = qword_4F1F628;
    v12 = *(_QWORD *)(qword_4F1F628 + 16);
  }
  *(_BYTE *)(*(_QWORD *)(v11 + 32) + v12) = 0;
  result = *(_QWORD *)(v11 + 32);
  ++*(_QWORD *)(v11 + 16);
  return result;
}

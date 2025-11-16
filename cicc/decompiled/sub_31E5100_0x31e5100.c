// Function: sub_31E5100
// Address: 0x31e5100
//
__int64 (*__fastcall sub_31E5100(__int64 a1))()
{
  __int64 v2; // rax
  char v3; // r13
  __int64 v4; // rax
  __int64 *v5; // r15
  __int64 *v6; // r12
  __int64 (*result)(); // rax
  __int64 (*v8)(); // rdi

  v2 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_501DA08);
  if ( !v2 )
    BUG();
  v3 = 0;
  v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_501DA08);
  v5 = *(__int64 **)(v4 + 176);
  v6 = &v5[*(unsigned int *)(v4 + 184)];
  if ( v6 == v5 )
    return (__int64 (*)())sub_2FC8EE0(a1 + 616);
  do
  {
    while ( 1 )
    {
      result = (__int64 (*)())sub_31E4CF0(a1, *v5);
      v8 = result;
      if ( result )
      {
        result = *(__int64 (**)())(*(_QWORD *)result + 32LL);
        if ( result != sub_31D49C0 )
        {
          result = (__int64 (*)())((__int64 (__fastcall *)(__int64 (*)(), __int64, __int64))result)(v8, a1 + 616, a1);
          if ( (_BYTE)result )
            break;
        }
      }
      ++v5;
      v3 = 1;
      if ( v6 == v5 )
        goto LABEL_9;
    }
    ++v5;
  }
  while ( v6 != v5 );
LABEL_9:
  if ( v3 )
    return (__int64 (*)())sub_2FC8EE0(a1 + 616);
  return result;
}

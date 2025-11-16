// Function: sub_852050
// Address: 0x852050
//
char *sub_852050()
{
  __int64 v0; // rcx
  __int64 v1; // r9
  size_t v2; // rsi
  char *result; // rax
  __int64 v4; // rbx
  unsigned __int64 ptr; // [rsp+8h] [rbp-18h] BYREF

  if ( fread(&ptr, 8u, 1u, qword_4F5FB48) != 1 )
    goto LABEL_10;
  v2 = ptr;
  result = ::ptr;
  if ( qword_4F5F850 < ptr )
  {
    v4 = qword_4F5F850 + 1024;
    if ( ptr >= qword_4F5F850 + 1024 )
      v4 = ptr;
    result = (char *)sub_822C60(::ptr, qword_4F5F850, v4, v0, qword_4F5F850, v1);
    v2 = ptr;
    qword_4F5F850 = v4;
    ::ptr = result;
    if ( !ptr )
      goto LABEL_4;
  }
  else if ( !ptr )
  {
LABEL_4:
    *result = 0;
    return result;
  }
  if ( fread(result, v2, 1u, qword_4F5FB48) != 1 )
LABEL_10:
    sub_851ED0();
  return ::ptr;
}

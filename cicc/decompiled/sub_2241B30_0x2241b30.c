// Function: sub_2241B30
// Address: 0x2241b30
//
int __fastcall sub_2241B30(_QWORD *a1, unsigned __int64 a2, size_t a3, const char *a4)
{
  unsigned __int64 v4; // rbx
  size_t v5; // rbx
  size_t v7; // rax
  size_t v8; // rdx
  size_t v9; // rbp
  int result; // eax
  __int64 v11; // rbx

  v4 = a1[1];
  if ( a2 > v4 )
    sub_222CF80("%s: __pos (which is %zu) > this->size() (which is %zu)", "basic_string::compare", a2, a1[1]);
  v5 = v4 - a2;
  if ( v5 > a3 )
    v5 = a3;
  v7 = strlen(a4);
  v8 = v5;
  v9 = v7;
  if ( v7 <= v5 )
    v8 = v7;
  if ( !v8 || (result = memcmp((const void *)(a2 + *a1), a4, v8)) == 0 )
  {
    v11 = v5 - v9;
    result = 0x7FFFFFFF;
    if ( v11 <= 0x7FFFFFFF )
    {
      result = 0x80000000;
      if ( v11 >= (__int64)0xFFFFFFFF80000000LL )
        return v11;
    }
  }
  return result;
}

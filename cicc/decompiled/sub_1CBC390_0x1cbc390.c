// Function: sub_1CBC390
// Address: 0x1cbc390
//
char __fastcall sub_1CBC390(__int64 a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // ebx
  int v4; // eax
  __int64 v5; // rax
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // rbx
  const char *v9; // r13
  __int64 v10; // rax
  size_t v11; // rdx
  const char *v12; // rsi
  size_t v13; // r12
  unsigned int v14; // eax

  v2 = sub_1CBC380(a1);
  if ( v2 != (unsigned int)sub_1CBC380(a2) )
  {
    v3 = sub_1CBC380(a1);
    LOBYTE(v4) = v3 > (unsigned int)sub_1CBC380(a2);
    return v4;
  }
  v5 = sub_1CBC2E0(a1);
  v6 = sub_1649960(v5);
  v8 = v7;
  v9 = v6;
  v10 = sub_1CBC2E0(a2);
  v12 = sub_1649960(v10);
  v13 = v11;
  if ( v8 <= v11 )
  {
    if ( !v8 || (v14 = memcmp(v9, v12, v8)) == 0 )
    {
      LOBYTE(v4) = 0;
      if ( v8 == v13 )
        return v4;
      goto LABEL_8;
    }
    return v14 >> 31;
  }
  LOBYTE(v4) = 0;
  if ( !v11 )
    return v4;
  v14 = memcmp(v9, v12, v11);
  if ( v14 )
    return v14 >> 31;
LABEL_8:
  LOBYTE(v4) = v8 < v13;
  return v4;
}

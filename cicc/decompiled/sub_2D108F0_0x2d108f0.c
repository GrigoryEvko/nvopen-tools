// Function: sub_2D108F0
// Address: 0x2d108f0
//
char __fastcall sub_2D108F0(__int64 a1, __int64 a2)
{
  int v2; // ebx
  unsigned int v3; // ebx
  int v4; // eax
  __int64 v5; // rax
  const char *v6; // rax
  size_t v7; // rdx
  size_t v8; // r13
  const char *v9; // r14
  __int64 v10; // rax
  const char *v11; // rax
  size_t v12; // rdx
  size_t v13; // rbx
  unsigned int v14; // eax

  v2 = sub_2D108E0(a1);
  if ( v2 == (unsigned int)sub_2D108E0(a2) )
  {
    v5 = sub_2D10810(a1);
    v6 = sub_BD5D20(v5);
    v8 = v7;
    v9 = v6;
    v10 = sub_2D10810(a2);
    v11 = sub_BD5D20(v10);
    v13 = v12;
    if ( v8 <= v12 )
      v12 = v8;
    if ( v12 && (v14 = memcmp(v9, v11, v12)) != 0 )
    {
      return v14 >> 31;
    }
    else
    {
      LOBYTE(v4) = v8 < v13;
      if ( v8 == v13 )
        LOBYTE(v4) = 0;
    }
  }
  else
  {
    v3 = sub_2D108E0(a1);
    LOBYTE(v4) = v3 > (unsigned int)sub_2D108E0(a2);
  }
  return v4;
}

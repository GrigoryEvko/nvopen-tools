// Function: sub_DF7D80
// Address: 0xdf7d80
//
__int64 __fastcall sub_DF7D80(__int64 a1, _BYTE *a2)
{
  unsigned int v2; // r12d
  const char *v4; // rax
  size_t v5; // rdx
  const char *v6; // r14
  size_t v7; // r13
  unsigned int v8; // eax

  v2 = 0;
  if ( (a2[33] & 0x20) != 0 )
    return v2;
  v2 = 1;
  if ( (a2[32] & 0xFu) - 7 <= 1 || (a2[7] & 0x10) == 0 )
    return v2;
  v4 = sub_BD5D20((__int64)a2);
  v6 = v4;
  v7 = v5;
  if ( v5 != 8 )
  {
    if ( v5 == 9 )
    {
      if ( *(_QWORD *)v4 == 0x6E67697379706F63LL && v4[8] == 102
        || *(_QWORD *)v4 == 0x6E67697379706F63LL && v4[8] == 108 )
      {
        return 0;
      }
    }
    else if ( v5 == 4 )
    {
      if ( *(_DWORD *)v4 == 1935827302 || *(_DWORD *)v4 == 1852403046 || *(_DWORD *)v4 == 2019650918 )
        return 0;
    }
    else if ( v5 == 5
           && (*(_DWORD *)v4 == 1935827302 && v4[4] == 102
            || *(_DWORD *)v4 == 1935827302 && v4[4] == 108
            || *(_DWORD *)v4 == 1852403046 && v4[4] == 102
            || *(_DWORD *)v4 == 1852403046 && v4[4] == 108
            || !memcmp(v4, "fmaxf", 5u)
            || !memcmp(v6, "fmaxl", 5u)) )
    {
      return 0;
    }
    goto LABEL_13;
  }
  if ( *(_QWORD *)v4 != 0x6E67697379706F63LL )
  {
LABEL_13:
    if ( sub_9691B0(v6, v7, "sin", 3) )
      return 0;
    if ( sub_9691B0(v6, v7, "sinf", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "sinl", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "cos", 3) )
      return 0;
    if ( sub_9691B0(v6, v7, "cosf", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "cosl", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "tan", 3) )
      return 0;
    if ( sub_9691B0(v6, v7, "tanf", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "tanl", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "asin", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "asinf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "asinl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "acos", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "acosf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "acosl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "atan", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "atanf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "atanl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "atan2", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "atan2f", 6) )
      return 0;
    if ( sub_9691B0(v6, v7, "atan2l", 6) )
      return 0;
    if ( sub_9691B0(v6, v7, "sinh", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "sinhf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "sinhl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "cosh", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "coshf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "coshl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "tanh", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "tanhf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "tanhl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "sqrt", 4) )
      return 0;
    if ( sub_9691B0(v6, v7, "sqrtf", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "sqrtl", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "exp10", 5) )
      return 0;
    if ( sub_9691B0(v6, v7, "exp10l", 6) )
      return 0;
    LOBYTE(v8) = sub_9691B0(v6, v7, "exp10f", 6);
    v2 = v8;
    if ( (_BYTE)v8 )
      return 0;
    if ( !sub_9691B0(v6, v7, "pow", 3)
      && !sub_9691B0(v6, v7, "powf", 4)
      && !sub_9691B0(v6, v7, "powl", 4)
      && !sub_9691B0(v6, v7, "exp2", 4)
      && !sub_9691B0(v6, v7, "exp2l", 5)
      && !sub_9691B0(v6, v7, "exp2f", 5)
      && !sub_9691B0(v6, v7, "floor", 5)
      && !sub_9691B0(v6, v7, "floorf", 6)
      && !sub_9691B0(v6, v7, "ceil", 4)
      && !sub_9691B0(v6, v7, "round", 5)
      && !sub_9691B0(v6, v7, "ffs", 3)
      && !sub_9691B0(v6, v7, "ffsl", 4)
      && !sub_9691B0(v6, v7, "abs", 3)
      && !sub_9691B0(v6, v7, "labs", 4) )
    {
      LOBYTE(v2) = !sub_9691B0(v6, v7, "llabs", 5);
    }
    return v2;
  }
  return 0;
}

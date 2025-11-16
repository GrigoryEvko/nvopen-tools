// Function: sub_3071FB0
// Address: 0x3071fb0
//
__int64 __fastcall sub_3071FB0(_BYTE *a1)
{
  unsigned int v1; // r12d
  const char *v3; // rax
  size_t v4; // rdx
  const char *v5; // r14
  size_t v6; // r13
  unsigned int v7; // eax

  v1 = 0;
  if ( (a1[33] & 0x20) != 0 )
    return v1;
  v1 = 1;
  if ( (a1[32] & 0xFu) - 7 <= 1 || (a1[7] & 0x10) == 0 )
    return v1;
  v3 = sub_BD5D20((__int64)a1);
  v5 = v3;
  v6 = v4;
  if ( v4 != 8 )
  {
    if ( v4 == 9 )
    {
      if ( *(_QWORD *)v3 == 0x6E67697379706F63LL && v3[8] == 102
        || *(_QWORD *)v3 == 0x6E67697379706F63LL && v3[8] == 108 )
      {
        return 0;
      }
    }
    else if ( v4 == 4 )
    {
      if ( *(_DWORD *)v3 == 1935827302 || *(_DWORD *)v3 == 1852403046 || *(_DWORD *)v3 == 2019650918 )
        return 0;
    }
    else if ( v4 == 5
           && (*(_DWORD *)v3 == 1935827302 && v3[4] == 102
            || *(_DWORD *)v3 == 1935827302 && v3[4] == 108
            || *(_DWORD *)v3 == 1852403046 && v3[4] == 102
            || *(_DWORD *)v3 == 1852403046 && v3[4] == 108
            || !memcmp(v3, "fmaxf", 5u)
            || !memcmp(v5, "fmaxl", 5u)) )
    {
      return 0;
    }
    goto LABEL_13;
  }
  if ( *(_QWORD *)v3 != 0x6E67697379706F63LL )
  {
LABEL_13:
    if ( sub_9691B0(v5, v6, "sin", 3) )
      return 0;
    if ( sub_9691B0(v5, v6, "sinf", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "sinl", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "cos", 3) )
      return 0;
    if ( sub_9691B0(v5, v6, "cosf", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "cosl", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "tan", 3) )
      return 0;
    if ( sub_9691B0(v5, v6, "tanf", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "tanl", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "asin", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "asinf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "asinl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "acos", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "acosf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "acosl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "atan", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "atanf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "atanl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "atan2", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "atan2f", 6) )
      return 0;
    if ( sub_9691B0(v5, v6, "atan2l", 6) )
      return 0;
    if ( sub_9691B0(v5, v6, "sinh", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "sinhf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "sinhl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "cosh", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "coshf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "coshl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "tanh", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "tanhf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "tanhl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "sqrt", 4) )
      return 0;
    if ( sub_9691B0(v5, v6, "sqrtf", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "sqrtl", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "exp10", 5) )
      return 0;
    if ( sub_9691B0(v5, v6, "exp10l", 6) )
      return 0;
    LOBYTE(v7) = sub_9691B0(v5, v6, "exp10f", 6);
    v1 = v7;
    if ( (_BYTE)v7 )
      return 0;
    if ( !sub_9691B0(v5, v6, "pow", 3)
      && !sub_9691B0(v5, v6, "powf", 4)
      && !sub_9691B0(v5, v6, "powl", 4)
      && !sub_9691B0(v5, v6, "exp2", 4)
      && !sub_9691B0(v5, v6, "exp2l", 5)
      && !sub_9691B0(v5, v6, "exp2f", 5)
      && !sub_9691B0(v5, v6, "floor", 5)
      && !sub_9691B0(v5, v6, "floorf", 6)
      && !sub_9691B0(v5, v6, "ceil", 4)
      && !sub_9691B0(v5, v6, "round", 5)
      && !sub_9691B0(v5, v6, "ffs", 3)
      && !sub_9691B0(v5, v6, "ffsl", 4)
      && !sub_9691B0(v5, v6, "abs", 3)
      && !sub_9691B0(v5, v6, "labs", 4) )
    {
      LOBYTE(v1) = !sub_9691B0(v5, v6, "llabs", 5);
    }
    return v1;
  }
  return 0;
}

// Function: sub_109CFD0
// Address: 0x109cfd0
//
__int64 __fastcall sub_109CFD0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // r15
  __int16 v4; // bx
  void *v5; // rax
  void *v6; // r14
  _QWORD *v7; // rdi
  __int64 v9; // rdx

  v4 = *(_WORD *)(a1 + 2);
  v5 = sub_C33340();
  v6 = v5;
  if ( v4 <= 0 )
  {
    v9 = -v4;
    if ( (void *)a2 == v5 )
      sub_C3C5A0((_QWORD *)(a1 + 8), (__int64)v5, v9);
    else
      sub_C36740(a1 + 8, a2, v9);
    v2 = (unsigned __int8 *)(a1 + 8);
    if ( v6 == *(void **)(a1 + 8) )
      sub_C3CCB0((__int64)v2);
    else
      sub_C34440(v2);
  }
  else
  {
    v7 = (_QWORD *)(a1 + 8);
    if ( (void *)a2 == v5 )
      sub_C3C5A0(v7, a2, v4);
    else
      sub_C36740((__int64)v7, a2, v4);
  }
  *(_WORD *)a1 = 257;
  return 257;
}

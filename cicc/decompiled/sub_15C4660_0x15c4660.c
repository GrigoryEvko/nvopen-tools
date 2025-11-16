// Function: sub_15C4660
// Address: 0x15c4660
//
__int64 __fastcall sub_15C4660(_QWORD *a1, _DWORD *a2)
{
  _QWORD *v2; // rcx
  __int64 result; // rax
  void *v4; // rsi
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // rdi

  v2 = (_QWORD *)a1[3];
  result = (__int64)a1;
  if ( a1[4] - (_QWORD)v2 > 0x18u && *v2 == 16 && v2[2] == 22 && v2[3] == 24 )
  {
    *a2 = v2[1];
    v4 = (void *)a1[3];
    if ( a1[4] - (_QWORD)v4 == 32 )
    {
      return 0;
    }
    else
    {
      v5 = a1[2];
      v6 = ((__int64)(a1[4] - (_QWORD)v4) >> 3) - 4;
      v7 = (__int64 *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v5 & 4) != 0 )
        v7 = (__int64 *)*v7;
      return sub_15C4420(v7, v4, v6, 0, 1);
    }
  }
  return result;
}

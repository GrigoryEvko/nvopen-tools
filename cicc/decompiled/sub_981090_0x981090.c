// Function: sub_981090
// Address: 0x981090
//
__int64 __fastcall sub_981090(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // r12
  __int64 i; // r13
  unsigned __int64 v12; // rdx
  __int64 v15; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v16[14]; // [rsp+18h] [rbp-38h] BYREF

  result = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 72) = 0;
  *(_OWORD *)(a1 + 8) = 0;
  *(_OWORD *)(a1 + 24) = 0;
  *(_OWORD *)(a1 + 40) = 0;
  *(_OWORD *)(a1 + 56) = 0;
  if ( (_BYTE)a4 )
  {
    result = sub_B2D620(a3, "no-builtins", 11);
    if ( (_BYTE)result )
    {
      *(_QWORD *)(a1 + 72) = 2047;
      *(_OWORD *)(a1 + 8) = -1;
      *(_OWORD *)(a1 + 24) = -1;
      *(_OWORD *)(a1 + 40) = -1;
      *(_OWORD *)(a1 + 56) = -1;
    }
    else
    {
      *(_QWORD *)v16 = *(_QWORD *)(a3 + 120);
      v15 = sub_A74680(v16);
      v10 = ((__int64 (__fastcall *)(__int64 *, const char *, __int64, __int64, __int64, __int64, __int64, __int64))sub_A73280)(
              &v15,
              "no-builtins",
              v6,
              v7,
              v8,
              v9,
              a3,
              a4);
      result = sub_A73290(&v15);
      for ( i = result; i != v10; v10 += 8 )
      {
        result = sub_A71840(v10);
        if ( (_BYTE)result )
        {
          result = sub_A71FD0(v10);
          if ( v12 > 0xA
            && *(_QWORD *)result == 0x746C6975622D6F6ELL
            && *(_WORD *)(result + 8) == 28265
            && *(_BYTE *)(result + 10) == 45 )
          {
            result = sub_980AF0(*(_QWORD *)a1, (_BYTE *)(result + 11), v12 - 11, v16);
            if ( (_BYTE)result )
            {
              if ( v16[0] > 0x20A )
                sub_222CF80("%s: __position (which is %zu) >= _Nb (which is %zu)", (char)"bitset::set");
              result = 1LL << SLOBYTE(v16[0]);
              *(_QWORD *)(a1 + 8 * ((unsigned __int64)v16[0] >> 6) + 8) |= 1LL << SLOBYTE(v16[0]);
            }
          }
        }
      }
    }
  }
  return result;
}

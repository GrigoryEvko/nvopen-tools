// Function: sub_211BAF0
// Address: 0x211baf0
//
__int64 __fastcall sub_211BAF0(__int64 *a1, __int64 a2, unsigned int a3)
{
  __int64 v3; // r13
  unsigned __int8 *v4; // rax
  __int64 v5; // r14
  __int64 v6; // r15
  unsigned int v7; // r9d
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rdx
  unsigned __int16 v11; // cx
  __int64 v13; // rdx
  _BYTE v14[16]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+10h] [rbp-40h]

  v3 = 40LL * a3;
  v4 = (unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v3 + *(_QWORD *)(a2 + 32)) + 40LL)
                         + 16LL * *(unsigned int *)(v3 + *(_QWORD *)(a2 + 32) + 8));
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  sub_1F40D10((__int64)v14, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v5, v6);
  v7 = 0;
  if ( (_BYTE)v5 == v14[8] )
  {
    LOBYTE(v7) = v6 != v15 && (_BYTE)v5 == 0;
    if ( (_BYTE)v7 )
      return 0;
    if ( (_BYTE)v5 && *(_QWORD *)(*a1 + 8 * v5 + 120) )
    {
      v8 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(a2 + 32) + v3) + 24LL);
      if ( (__int16)v8 <= 47 )
      {
        if ( (__int16)v8 <= 7 || (v13 = 0xC00000000900LL, !_bittest64(&v13, v8)) )
        {
LABEL_8:
          v11 = *(_WORD *)(a2 + 24);
          if ( v11 <= 0x2Fu )
            LOBYTE(v7) = ((1LL << v11) & 0x800000000900LL) != 0;
          return v7;
        }
      }
      else
      {
        v9 = (unsigned int)(v8 - 101);
        if ( (unsigned __int16)v9 > 0x3Eu )
          goto LABEL_8;
        v10 = 0x6200000A00000001LL;
        if ( !_bittest64(&v10, v9) )
          goto LABEL_8;
      }
      return 1;
    }
  }
  return v7;
}

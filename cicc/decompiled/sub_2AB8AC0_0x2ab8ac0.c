// Function: sub_2AB8AC0
// Address: 0x2ab8ac0
//
__int64 __fastcall sub_2AB8AC0(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  unsigned __int64 v3; // rbx
  int v4; // r14d
  __int64 v6; // r14
  unsigned __int64 v7; // rax
  char v8; // dl
  __int64 v9; // [rsp+18h] [rbp-38h]

  v3 = HIDWORD(a3);
  HIDWORD(v9) = HIDWORD(a3);
  if ( *(_QWORD *)(*(_QWORD *)a2 + 8LL) != *(_QWORD *)(*(_QWORD *)a2 + 16LL) )
  {
    v4 = a3;
    if ( (_DWORD)a3 )
    {
      if ( BYTE4(a3) && !(unsigned __int8)sub_DFE610(*(_QWORD *)(a2 + 32)) && !byte_500DD68 )
      {
        sub_2AB8760(
          (__int64)"Scalable vectorization requested but not supported by the target",
          64,
          "the scalable user-specified vectorization width for outer-loop vectorization cannot be used because the target"
          " does not support scalable vectors.",
          0x91u,
          (__int64)"ScalableVFUnfeasible",
          20,
          *(__int64 **)(a2 + 80),
          *(_QWORD *)a2,
          0);
        *(_QWORD *)a1 = 1;
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        *(_QWORD *)(a1 + 32) = 0;
        *(_QWORD *)(a1 + 40) = 0;
        return a1;
      }
    }
    else
    {
      v6 = *(_QWORD *)(a2 + 32);
      v3 = (unsigned __int64)sub_2AB4370(*(_QWORD *)(a2 + 48)) >> 32;
      sub_DFE640(v6);
      v7 = sub_DFB1B0(v6) / v3;
      LOBYTE(v3) = v8;
      v4 = v7;
      if ( byte_500D208 )
      {
        if ( v8 != 1 && (_DWORD)v7 == 1 )
        {
          LOBYTE(v3) = 0;
          v4 = 4;
        }
        else if ( !(_DWORD)v7 )
        {
          LOBYTE(v3) = 0;
          v4 = 4;
        }
      }
    }
    LODWORD(v9) = v4;
    BYTE4(v9) = v3;
    sub_2BF4260(a2, v9);
    if ( !byte_500D208 )
    {
      *(_BYTE *)(a1 + 44) = 0;
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = v9;
      *(_DWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      *(_DWORD *)(a1 + 32) = 0;
      *(_DWORD *)(a1 + 40) = 0;
      return a1;
    }
  }
  *(_QWORD *)a1 = 1;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  return a1;
}

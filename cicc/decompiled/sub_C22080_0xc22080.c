// Function: sub_C22080
// Address: 0xc22080
//
__int64 __fastcall sub_C22080(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v4; // rdx
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 (__fastcall ***v11)(); // rax
  _QWORD v12[2]; // [rsp+0h] [rbp-40h] BYREF
  char v13; // [rsp+10h] [rbp-30h]

  sub_C21E40((__int64)v12, a2);
  if ( (v13 & 1) != 0 && (v4 = v12[1], LODWORD(v12[0])) )
  {
    *(_DWORD *)a1 = v12[0];
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_QWORD *)(a1 + 8) = v4;
    return a1;
  }
  else
  {
    v6 = a2[31];
    v7 = v12[0];
    if ( v12[0] >= 0xCCCCCCCCCCCCCCCDLL * ((a2[32] - v6) >> 3) )
    {
      *(_BYTE *)(a1 + 16) |= 1u;
      v11 = sub_C1AFD0();
      *(_DWORD *)a1 = 8;
      *(_QWORD *)(a1 + 8) = v11;
      return a1;
    }
    else
    {
      if ( a3 )
      {
        *a3 = v12[0];
        v6 = a2[31];
      }
      *(_BYTE *)(a1 + 16) &= ~1u;
      v8 = (__int64 *)(v6 + 40 * v7);
      v9 = *v8;
      v10 = *((unsigned int *)v8 + 2);
      *(_QWORD *)a1 = v9;
      *(_QWORD *)(a1 + 8) = v10;
      return a1;
    }
  }
}

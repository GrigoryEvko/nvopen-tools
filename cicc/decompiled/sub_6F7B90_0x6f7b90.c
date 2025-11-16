// Function: sub_6F7B90
// Address: 0x6f7b90
//
__int64 __fastcall sub_6F7B90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, _QWORD *a6)
{
  unsigned __int8 v7; // r12
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // r15
  __int64 result; // rax
  __int64 v15; // [rsp+0h] [rbp-50h]
  int v16; // [rsp+Ch] [rbp-44h]
  int v17; // [rsp+18h] [rbp-38h] BYREF
  _DWORD v18[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v16 = a5;
  if ( (_BYTE)a3 == 119 )
  {
    sub_6E6260(a6);
  }
  else
  {
    v7 = a3;
    if ( (_BYTE)a3 == 91 )
      v9 = sub_6F7180((const __m128i *)a1, 0, a3, a4, a5, (__int64)a6);
    else
      v9 = sub_6F6F40((const __m128i *)a1, 0, a3, a4, a5, (__int64)a6);
    v15 = v9;
    *(_QWORD *)(v9 + 16) = sub_6F6F40((const __m128i *)a2, 0, v9, v10, v11, v12);
    v13 = sub_73DBF0(v7, a4, v15);
    if ( unk_4D04810 )
    {
      sub_737810(v7, &v17, v18);
      *(_BYTE *)(v13 + 60) = *(_BYTE *)(v13 + 60) & 0xFC | v17 & 1 | (2 * (v18[0] & 1));
    }
    sub_6E70E0((__int64 *)v13, (__int64)a6);
    if ( v16 )
    {
      *(_BYTE *)(v13 + 25) |= 1u;
      if ( v7 == 91 )
        *(_BYTE *)(v13 + 58) |= 1u;
      sub_6E6A20((__int64)a6);
    }
  }
  *(_QWORD *)((char *)a6 + 68) = *(_QWORD *)(a1 + 68);
  *(_QWORD *)((char *)a6 + 76) = *(_QWORD *)(a1 + 76);
  result = *(_QWORD *)(a2 + 76);
  *(_QWORD *)((char *)a6 + 76) = result;
  return result;
}

// Function: sub_7736E0
// Address: 0x7736e0
//
__int64 __fastcall sub_7736E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, FILE *a6)
{
  __int64 result; // rax
  _QWORD *v8; // rax
  __int64 v11; // rdx
  __int64 v12; // [rsp+8h] [rbp-38h]
  unsigned int v13; // [rsp+8h] [rbp-38h]

  result = dword_4D04880;
  if ( dword_4D04880 )
  {
    v8 = *(_QWORD **)(a1 + 16);
    if ( (unsigned int)(0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24))) <= 0x2F )
    {
      v12 = a5;
      sub_772E70((_QWORD *)(a1 + 16));
      v8 = *(_QWORD **)(a1 + 16);
      a5 = v12;
    }
    *(_QWORD *)(a1 + 16) = v8 + 6;
    v11 = *(_QWORD *)(a1 + 48);
    v8[1] = a2;
    *v8 = v11;
    v8[2] = a3;
    v8[3] = a4;
    v8[4] = a5;
    v8[5] = a6;
    *(_QWORD *)(a1 + 48) = v8;
    return 1;
  }
  else if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    v13 = dword_4D04880;
    sub_6855B0(0xAABu, a6, (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return v13;
  }
  return result;
}

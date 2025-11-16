// Function: sub_14586F0
// Address: 0x14586f0
//
__int64 __fastcall sub_14586F0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *i; // rdx
  __int64 result; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 56) = 128;
  v4 = (_QWORD *)sub_22077B0(6144);
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 40) = v4;
  for ( i = &v4[6 * *(unsigned int *)(a1 + 56)]; i != v4; v4 += 6 )
  {
    if ( v4 )
    {
      v4[2] = 0;
      v4[3] = -8;
      *v4 = &unk_49EC740;
      v4[1] = 2;
      v4[4] = 0;
    }
  }
  *(_QWORD *)(a1 + 112) = a2;
  *(_QWORD *)(a1 + 120) = a3;
  *(_BYTE *)(a1 + 96) = 0;
  *(_BYTE *)(a1 + 105) = 1;
  result = sub_14585E0(a1 + 128);
  *(_DWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 0;
  return result;
}

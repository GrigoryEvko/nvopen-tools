// Function: sub_C349F0
// Address: 0xc349f0
//
__int64 __fastcall sub_C349F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  char v3; // cl
  __int64 v4; // rbx
  __int64 v5; // r14
  __int64 v6; // rdx

  v2 = *(unsigned __int8 *)(a2 + 20);
  v3 = *(_BYTE *)(a2 + 20) & 7;
  if ( v3 == 1 )
  {
    v4 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 2046;
    v6 = *(_QWORD *)sub_C33930(a2) & 0xFFFFFFFFFFFFFLL;
    v2 = *(unsigned __int8 *)(a2 + 20);
  }
  else if ( v3 == 3 )
  {
    v6 = 0;
    v4 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) - 1;
  }
  else if ( v3 )
  {
    v4 = *(_DWORD *)(a2 + 16) + (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 1022;
    v5 = *(_QWORD *)sub_C33930(a2);
    if ( (int)v4 == 1 )
      v4 = (*(_QWORD *)sub_C33930(a2) >> 52) & 1LL;
    v6 = v5 & 0xFFFFFFFFFFFFFLL;
    v2 = *(unsigned __int8 *)(a2 + 20);
  }
  else
  {
    v4 = (*(_QWORD *)a2 != (_QWORD)&unk_3F65660) + 2046;
    v6 = 0;
  }
  LOBYTE(v2) = (unsigned __int8)v2 >> 3;
  *(_DWORD *)(a1 + 8) = 64;
  *(_QWORD *)a1 = (v4 << 52) & 0x7FF0000000000000LL | v6 | (v2 << 63);
  return a1;
}

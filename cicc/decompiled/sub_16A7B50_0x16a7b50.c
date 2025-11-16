// Function: sub_16A7B50
// Address: 0x16a7b50
//
__int64 __fastcall sub_16A7B50(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v4; // r13d
  unsigned __int64 v5; // r14
  __int64 v6; // rax
  _QWORD *v8; // [rsp+8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  v5 = 0xFFFFFFFFFFFFFFFFLL >> -*(_BYTE *)(a2 + 8);
  if ( v4 > 0x40 )
  {
    v8 = (_QWORD *)sub_2207820(8 * (((unsigned __int64)v4 + 63) >> 6));
    sub_16A7AB0(v8, *(_QWORD *)a2, *a3, ((unsigned __int64)*(unsigned int *)(a2 + 8) + 63) >> 6);
    *(_DWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v8;
    v8[(unsigned int)(((unsigned __int64)v4 + 63) >> 6) - 1] &= v5;
  }
  else
  {
    v6 = *a3 * *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v6 & v5;
  }
  return a1;
}

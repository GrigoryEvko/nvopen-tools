// Function: sub_2AC7960
// Address: 0x2ac7960
//
__int64 __fastcall sub_2AC7960(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // esi
  int v5; // eax
  int v6; // eax
  __int64 result; // rax
  _QWORD v8[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v8[0] = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= 3 * v4 )
  {
    v4 *= 2;
    goto LABEL_9;
  }
  if ( v4 - *(_DWORD *)(a1 + 20) - v6 <= v4 >> 3 )
  {
LABEL_9:
    sub_2AC77A0(a1, v4);
    sub_2ABE410(a1, a2, v8);
    v6 = *(_DWORD *)(a1 + 16) + 1;
  }
  *(_DWORD *)(a1 + 16) = v6;
  result = v8[0];
  if ( *(_QWORD *)v8[0] != -4096 || *(_DWORD *)(v8[0] + 8LL) != -1 || !*(_BYTE *)(v8[0] + 12LL) )
    --*(_DWORD *)(a1 + 20);
  return result;
}

// Function: sub_B054A0
// Address: 0xb054a0
//
__int64 __fastcall sub_B054A0(__int64 a1, int a2, __int64 a3)
{
  unsigned int v5; // esi
  int v6; // eax
  _QWORD *v7; // rdx
  int v8; // eax
  __int64 v9; // [rsp+8h] [rbp-28h] BYREF
  _QWORD *v10; // [rsp+10h] [rbp-20h] BYREF
  _QWORD *v11; // [rsp+18h] [rbp-18h] BYREF

  v9 = a1;
  if ( a2 )
  {
    if ( a2 == 1 )
      sub_B95A20(v9);
    return v9;
  }
  if ( (unsigned __int8)sub_AFD040(a3, &v9, &v10) )
    return v9;
  v5 = *(_DWORD *)(a3 + 24);
  v6 = *(_DWORD *)(a3 + 16);
  v7 = v10;
  ++*(_QWORD *)a3;
  v8 = v6 + 1;
  v11 = v7;
  if ( 4 * v8 >= 3 * v5 )
  {
    v5 *= 2;
    goto LABEL_12;
  }
  if ( v5 - *(_DWORD *)(a3 + 20) - v8 <= v5 >> 3 )
  {
LABEL_12:
    sub_B05150(a3, v5);
    sub_AFD040(a3, &v9, &v11);
    v7 = v11;
    v8 = *(_DWORD *)(a3 + 16) + 1;
  }
  *(_DWORD *)(a3 + 16) = v8;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a3 + 20);
  *v7 = v9;
  return v9;
}

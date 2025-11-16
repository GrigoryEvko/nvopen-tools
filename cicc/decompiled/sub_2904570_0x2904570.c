// Function: sub_2904570
// Address: 0x2904570
//
unsigned __int64 __fastcall sub_2904570(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  unsigned int v4; // esi
  int v5; // eax
  __int64 v6; // rcx
  int v7; // eax
  __int64 v8; // [rsp+0h] [rbp-30h] BYREF
  __int64 v9; // [rsp+8h] [rbp-28h] BYREF
  __int64 v10; // [rsp+10h] [rbp-20h] BYREF
  int v11; // [rsp+18h] [rbp-18h]

  v2 = *a2;
  v11 = 0;
  v10 = v2;
  if ( !(unsigned __int8)sub_22B1A50(a1, &v10, &v8) )
  {
    v4 = *(_DWORD *)(a1 + 24);
    v5 = *(_DWORD *)(a1 + 16);
    v6 = v8;
    ++*(_QWORD *)a1;
    v7 = v5 + 1;
    v9 = v6;
    if ( 4 * v7 >= 3 * v4 )
    {
      v4 *= 2;
    }
    else if ( v4 - *(_DWORD *)(a1 + 20) - v7 > v4 >> 3 )
    {
      goto LABEL_5;
    }
    sub_D39D40(a1, v4);
    sub_22B1A50(a1, &v10, &v9);
    v6 = v9;
    v7 = *(_DWORD *)(a1 + 16) + 1;
LABEL_5:
    *(_DWORD *)(a1 + 16) = v7;
    if ( *(_QWORD *)v6 != -4096 )
      --*(_DWORD *)(a1 + 20);
    *(_QWORD *)v6 = v10;
    *(_DWORD *)(v6 + 8) = v11;
    BUG();
  }
  return *(_QWORD *)(a1 + 32) + ((unsigned __int64)*(unsigned int *)(v8 + 8) << 6) + 8;
}

// Function: sub_B494D0
// Address: 0xb494d0
//
__int64 __fastcall sub_B494D0(__int64 a1, int a2)
{
  __int64 v2; // rax
  int v4; // [rsp+4h] [rbp-2Ch] BYREF
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_A74390((__int64 *)(a1 + 72), a2, &v4) )
    return *(_QWORD *)(a1 + 32 * ((unsigned int)(v4 - 1) - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v2 = *(_QWORD *)(a1 - 32);
  if ( v2
    && !*(_BYTE *)v2
    && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80)
    && (v5[0] = *(_QWORD *)(v2 + 120), (unsigned __int8)sub_A74390(v5, a2, &v4)) )
  {
    return *(_QWORD *)(a1 + 32 * ((unsigned int)(v4 - 1) - (unsigned __int64)(*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  }
  else
  {
    return 0;
  }
}

// Function: sub_29A2380
// Address: 0x29a2380
//
__int64 __fastcall sub_29A2380(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  char v4; // al
  __int64 result; // rax
  __int64 v6; // r13
  const char *v7; // rax
  __int64 v8; // rdx

  sub_B2CA40(a2, 0);
  v4 = *(_BYTE *)(a2 + 32);
  *(_BYTE *)(a2 + 32) = v4 & 0xF0;
  if ( (v4 & 0x30) != 0 )
    *(_BYTE *)(a2 + 33) |= 0x40u;
  if ( *(_QWORD *)(a2 + 48) )
  {
    result = *(unsigned int *)(a1 + 312);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 316) )
    {
      sub_C8D5F0(a1 + 304, (const void *)(a1 + 320), result + 1, 8u, v2, v3);
      result = *(unsigned int *)(a1 + 312);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 312);
  }
  else
  {
    result = *(unsigned int *)(a1 + 168);
    if ( result + 1 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), result + 1, 8u, v2, v3);
      result = *(unsigned int *)(a1 + 168);
    }
    *(_QWORD *)(*(_QWORD *)(a1 + 160) + 8 * result) = a2;
    ++*(_DWORD *)(a1 + 168);
  }
  v6 = *(_QWORD *)(a1 + 480);
  if ( v6 )
  {
    v7 = sub_BD5D20(a2);
    return sub_BBB260(v6, a2, (__int64)v7, v8);
  }
  return result;
}

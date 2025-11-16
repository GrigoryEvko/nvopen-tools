// Function: sub_C36E40
// Address: 0xc36e40
//
__int64 __fastcall sub_C36E40(__int64 a1, __int64 *a2)
{
  __int64 *v2; // r13
  int v3; // eax
  __int64 result; // rax

  if ( *((_DWORD *)a2 + 2) > 0x40u )
    a2 = (__int64 *)*a2;
  v2 = (__int64 *)*a2;
  sub_C337F0((_QWORD *)a1, (__int64)&unk_3F65660);
  *(_BYTE *)(a1 + 20) &= ~8u;
  *(_QWORD *)sub_C33900(a1) = 1;
  v3 = *(_BYTE *)(a1 + 20) & 0xF8;
  if ( v2 == (__int64 *)255 )
  {
    *(_BYTE *)(a1 + 20) = v3 | 1;
    result = sub_C36030((unsigned int **)a1);
    *(_DWORD *)(a1 + 16) = result;
  }
  else
  {
    result = v3 | 2u;
    *(_DWORD *)(a1 + 16) = (unsigned __int8)v2 - 127;
    *(_BYTE *)(a1 + 20) = result;
  }
  return result;
}

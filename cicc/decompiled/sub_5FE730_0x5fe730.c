// Function: sub_5FE730
// Address: 0x5fe730
//
__int64 __fastcall sub_5FE730(__int64 *a1, int a2)
{
  char v2; // al
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  _BYTE v6[112]; // [rsp+0h] [rbp-2E0h] BYREF
  _QWORD v7[70]; // [rsp+70h] [rbp-270h] BYREF
  char v8; // [rsp+2A0h] [rbp-40h]

  sub_5E4C60((__int64)v7, (_QWORD *)(*a1 + 64));
  v2 = v8;
  v8 |= 2u;
  if ( (*((_BYTE *)a1 + 9) & 0x10) == 0 )
    v8 = v2 | 6;
  sub_87E3B0(v6);
  result = sub_5FE480(a1, (__int64)v7, (__int64)v6, 0);
  if ( a2 )
  {
    result = dword_4D04464;
    if ( dword_4D04464 )
    {
      v4 = v7[0];
      v5 = *(_QWORD *)(v7[0] + 88LL);
      *(_BYTE *)(v7[0] + 81LL) |= 2u;
      *(_BYTE *)(v5 + 206) |= 0x10u;
      result = *(_QWORD *)(v4 + 88);
      *(_BYTE *)(result + 193) |= 0x20u;
    }
  }
  return result;
}

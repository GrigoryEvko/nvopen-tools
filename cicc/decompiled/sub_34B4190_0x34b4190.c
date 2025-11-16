// Function: sub_34B4190
// Address: 0x34b4190
//
__int64 __fastcall sub_34B4190(_QWORD *a1, unsigned int a2)
{
  __int64 v2; // r12
  _BYTE *v3; // rsi
  __int64 v4; // rax
  unsigned int v5; // r8d
  _DWORD v7[5]; // [rsp+Ch] [rbp-14h] BYREF

  v2 = a2;
  v3 = (_BYTE *)a1[2];
  v4 = (__int64)&v3[-a1[1]] >> 2;
  v7[0] = v4;
  if ( v3 == (_BYTE *)a1[3] )
  {
    sub_B8BBF0((__int64)(a1 + 1), v3, v7);
    v5 = v7[0];
  }
  else
  {
    v5 = v4;
    if ( v3 )
    {
      *(_DWORD *)v3 = v4;
      v3 = (_BYTE *)a1[2];
    }
    a1[2] = v3 + 4;
  }
  *(_DWORD *)(a1[4] + 4 * v2) = v5;
  return v5;
}

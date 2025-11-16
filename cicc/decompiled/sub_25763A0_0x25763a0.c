// Function: sub_25763A0
// Address: 0x25763a0
//
__int64 __fastcall sub_25763A0(__int64 a1, _BYTE *a2, __int64 a3, unsigned __int64 a4)
{
  unsigned int v4; // r12d
  unsigned int v6; // eax
  char v7; // [rsp+Eh] [rbp-32h] BYREF
  char v8; // [rsp+Fh] [rbp-31h] BYREF
  unsigned __int64 v9; // [rsp+10h] [rbp-30h] BYREF
  unsigned int v10; // [rsp+18h] [rbp-28h]

  v4 = 0;
  v7 = 0;
  v8 = 0;
  sub_254BC20((__int64)&v9, a2, a3, a4, &v7, &v8);
  if ( !v8 )
  {
    v4 = *(unsigned __int8 *)(a1 + 105);
    if ( !v7 )
    {
      if ( (_BYTE)v4 )
      {
        sub_2575FB0((_DWORD *)(a1 + 112), (const void **)&v9);
        v6 = *(_DWORD *)(a1 + 152);
        if ( v6 < unk_4FEF868 )
        {
          v4 = *(unsigned __int8 *)(a1 + 105);
          *(_BYTE *)(a1 + 288) &= v6 == 0;
        }
        else
        {
          v4 = *(unsigned __int8 *)(a1 + 104);
          *(_BYTE *)(a1 + 105) = v4;
        }
      }
    }
  }
  if ( v10 > 0x40 && v9 )
    j_j___libc_free_0_0(v9);
  return v4;
}

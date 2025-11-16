// Function: sub_B93600
// Address: 0xb93600
//
unsigned __int64 __fastcall sub_B93600(__int64 a1)
{
  _BYTE *v2; // rax
  unsigned __int8 v3; // dl
  _BYTE *v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 v9; // [rsp+8h] [rbp-18h]

  v2 = *(_BYTE **)a1;
  v8 = 0;
  v9 = 0;
  if ( v2 && *v2 == 14 )
  {
    v3 = *(v2 - 16);
    v4 = (v3 & 2) != 0 ? (_BYTE *)*((_QWORD *)v2 - 4) : &v2[-8 * ((v3 >> 2) & 0xF) - 16];
    v5 = *((_QWORD *)v4 + 7);
    if ( v5 )
    {
      v8 = sub_B91420(v5);
      v9 = v6;
    }
  }
  if ( (*(_BYTE *)(a1 + 76) & 8) == 0 && *(_QWORD *)(a1 + 16) && *(_QWORD *)a1 && **(_BYTE **)a1 == 14 )
    return sub_AFA7A0((__int64 *)(a1 + 16), &v8);
  else
    return sub_AFA420((__int64 *)(a1 + 8), &v8, (__int64 *)(a1 + 24), (__int64 *)(a1 + 40), (int *)(a1 + 32));
}

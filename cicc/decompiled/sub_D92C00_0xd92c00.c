// Function: sub_D92C00
// Address: 0xd92c00
//
__int64 __fastcall sub_D92C00(__int64 a1, __int64 a2, __int64 a3)
{
  const void **v3; // rbx
  char v4; // al
  unsigned int v5; // r14d
  unsigned int v6; // eax
  bool v7; // cc
  unsigned int v8; // eax
  __int64 v10; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v11; // [rsp+18h] [rbp-48h]
  __int64 v12; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v13; // [rsp+28h] [rbp-38h]

  v3 = (const void **)a3;
  v4 = *(_BYTE *)(a3 + 16);
  if ( !*(_BYTE *)(a2 + 16) )
  {
    if ( !v4 )
    {
      *(_BYTE *)(a1 + 16) = 0;
      return a1;
    }
    a2 = a3;
    goto LABEL_18;
  }
  if ( !v4 )
  {
LABEL_18:
    v8 = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 8) = v8;
    if ( v8 > 0x40 )
      sub_C43780(a1, (const void **)a2);
    else
      *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_BYTE *)(a1 + 16) = 1;
    return a1;
  }
  v5 = *(_DWORD *)(a2 + 8);
  if ( *(_DWORD *)(a3 + 8) >= v5 )
    v5 = *(_DWORD *)(a3 + 8);
  sub_C44830((__int64)&v10, (_DWORD *)a2, v5);
  sub_C44830((__int64)&v12, v3, v5);
  if ( (int)sub_C4C880((__int64)&v10, (__int64)&v12) < 0 )
    v3 = (const void **)a2;
  v6 = *((_DWORD *)v3 + 2);
  *(_DWORD *)(a1 + 8) = v6;
  if ( v6 > 0x40 )
    sub_C43780(a1, v3);
  else
    *(_QWORD *)a1 = *v3;
  v7 = v13 <= 0x40;
  *(_BYTE *)(a1 + 16) = 1;
  if ( !v7 && v12 )
    j_j___libc_free_0_0(v12);
  if ( v11 > 0x40 && v10 )
    j_j___libc_free_0_0(v10);
  return a1;
}

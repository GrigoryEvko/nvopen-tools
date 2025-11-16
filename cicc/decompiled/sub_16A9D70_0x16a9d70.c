// Function: sub_16A9D70
// Address: 0x16a9d70
//
__int64 __fastcall sub_16A9D70(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // ecx
  int v5; // eax
  int v6; // ecx
  unsigned int v7; // r15d
  unsigned __int64 v8; // r14
  int v9; // eax
  unsigned int v10; // r15d
  unsigned __int64 v11; // r10
  unsigned int v12; // r15d
  int v13; // eax
  unsigned __int64 v14; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // [rsp+8h] [rbp-48h]
  int v19; // [rsp+Ch] [rbp-44h]
  int v20; // [rsp+Ch] [rbp-44h]
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  int v22; // [rsp+18h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 8);
  if ( v4 <= 0x40 )
  {
    v16 = *(_QWORD *)a2 / *(_QWORD *)a3;
    *(_DWORD *)(a1 + 8) = v4;
    *(_QWORD *)a1 = v16 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v4);
    return a1;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v5 = sub_16A57B0(a2);
  v6 = v19;
  v7 = *(_DWORD *)(a3 + 8);
  v8 = ((unsigned __int64)(unsigned int)(v19 - v5) + 63) >> 6;
  if ( v7 > 0x40 )
  {
    v9 = sub_16A57B0(a3);
    v6 = v19;
    v10 = v7 - v9;
    v11 = ((unsigned __int64)v10 + 63) >> 6;
    if ( !v8 )
    {
LABEL_12:
      *(_DWORD *)(a1 + 8) = v6;
      sub_16A4EF0(a1, 0, 0);
      return a1;
    }
    goto LABEL_4;
  }
  if ( *(_QWORD *)a3 )
  {
    _BitScanReverse64(&v14, *(_QWORD *)a3);
    LODWORD(v11) = 1;
    v10 = 64 - (v14 ^ 0x3F);
    if ( !v8 )
      goto LABEL_12;
LABEL_4:
    if ( v10 == 1 )
    {
      *(_DWORD *)(a1 + 8) = v6;
      sub_16A4FD0(a1, (const void **)a2);
      return a1;
    }
    v12 = v8;
    if ( (unsigned int)v8 < (unsigned int)v11 )
      goto LABEL_12;
    goto LABEL_6;
  }
  v12 = ((unsigned __int64)(unsigned int)(v19 - v5) + 63) >> 6;
  LODWORD(v11) = 0;
  if ( !v8 )
    goto LABEL_12;
LABEL_6:
  v18 = v11;
  v20 = v6;
  v13 = sub_16A9900(a2, (unsigned __int64 *)a3);
  v6 = v20;
  if ( v13 < 0 )
    goto LABEL_12;
  if ( sub_16A5220(a2, (const void **)a3) )
  {
    *(_DWORD *)(a1 + 8) = v20;
    sub_16A4EF0(a1, 1, 0);
  }
  else if ( v12 == 1 )
  {
    v17 = **(_QWORD **)a2 / **(_QWORD **)a3;
    *(_DWORD *)(a1 + 8) = v20;
    sub_16A4EF0(a1, v17, 0);
  }
  else
  {
    v22 = v20;
    sub_16A4EF0((__int64)&v21, 0, 0);
    sub_16A6110(*(__int64 **)a2, v12, *(__int64 **)a3, v18, v21, 0);
    *(_DWORD *)(a1 + 8) = v22;
    *(_QWORD *)a1 = v21;
  }
  return a1;
}

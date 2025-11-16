// Function: sub_2B2CC60
// Address: 0x2b2cc60
//
__int64 __fastcall sub_2B2CC60(__int64 a1, unsigned int a2)
{
  __int64 v2; // r12
  char v3; // al
  unsigned __int8 v5; // dl
  int v6; // eax
  __int64 v7; // rcx
  __int64 *v8; // rcx
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // r14
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 *v16; // rdx
  __int64 v17; // r14
  __int64 v18; // rax
  int v19; // edx
  __int64 v21; // [rsp+30h] [rbp-40h] BYREF
  char v22[56]; // [rsp+38h] [rbp-38h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2);
  v3 = *(_BYTE *)v2;
  if ( *(_BYTE *)v2 == 13 )
    return 0;
  v5 = *(_BYTE *)(**(_QWORD **)(a1 + 8) + 8LL);
  if ( v5 <= 3u || v5 == 5 )
  {
    if ( v3 == 86 )
      goto LABEL_11;
  }
  else if ( v3 == 86 )
  {
LABEL_11:
    if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
      v8 = *(__int64 **)(v2 - 8);
    else
      v8 = (__int64 *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
    if ( (unsigned __int8)(*(_BYTE *)*v8 - 82) <= 1u )
    {
      v6 = sub_B53900(*v8);
      goto LABEL_6;
    }
LABEL_15:
    v7 = *(_QWORD *)(a1 + 24);
LABEL_16:
    v9 = 16;
    if ( v5 > 3u && v5 != 5 && (v5 & 0xFD) != 4 )
      v9 = 42;
    *(_DWORD *)v7 = v9;
    *(_BYTE *)(v7 + 4) = 0;
    v10 = *(_QWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    *(_DWORD *)v11 = *(_DWORD *)v10;
    *(_BYTE *)(v11 + 4) = *(_BYTE *)(v10 + 4);
    goto LABEL_21;
  }
  if ( (unsigned __int8)(v3 - 82) > 1u )
    goto LABEL_15;
  v6 = sub_B53900(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 32LL) + 8LL * a2));
LABEL_6:
  if ( **(_DWORD **)(a1 + 16) != v6 )
  {
    v7 = *(_QWORD *)(a1 + 24);
    if ( v6 != *(_DWORD *)v7 )
    {
      v5 = *(_BYTE *)(**(_QWORD **)(a1 + 8) + 8LL);
      goto LABEL_16;
    }
  }
LABEL_21:
  v12 = *(_QWORD *)(a1 + 32);
  v13 = *(__int64 **)(v12 + 3296);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v14 = *(_QWORD *)(v2 - 8);
  else
    v14 = v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF);
  *(_QWORD *)v22 = *(_QWORD *)(v14 + 32);
  sub_2B2BBE0(v12, v22, 1);
  v15 = *(_QWORD *)(a1 + 32);
  if ( (*(_BYTE *)(v2 + 7) & 0x40) != 0 )
    v16 = *(__int64 **)(v2 - 8);
  else
    v16 = (__int64 *)(v2 - 32LL * (*(_DWORD *)(v2 + 4) & 0x7FFFFFF));
  v21 = *v16;
  sub_2B2BBE0(v15, (char *)&v21, 1);
  sub_BCB2A0(*(_QWORD **)(*(_QWORD *)(a1 + 32) + 3440LL));
  v17 = sub_DFD2D0(
          v13,
          (unsigned int)**(unsigned __int8 **)(**(_QWORD **)(a1 + 40) + 416LL) - 29,
          **(_QWORD **)(a1 + 48));
  v18 = sub_2B21F70(*(_QWORD *)(a1 + 64), **(_QWORD **)(a1 + 48), v2);
  if ( !v19 )
    return v18;
  return v17;
}

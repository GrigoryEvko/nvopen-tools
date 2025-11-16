// Function: sub_1D672E0
// Address: 0x1d672e0
//
void __fastcall sub_1D672E0(__int64 a1)
{
  int v1; // ebx
  __int64 v2; // r12
  _QWORD *v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 i; // rcx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax
  bool v10; // zf
  unsigned int v11; // eax
  int v12; // r12d
  unsigned int v13; // ebx
  unsigned int v14; // eax
  unsigned int v15; // eax
  _QWORD v16[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v17; // [rsp+18h] [rbp-68h]
  __int64 v18; // [rsp+20h] [rbp-60h]
  void *v19; // [rsp+30h] [rbp-50h]
  _QWORD v20[2]; // [rsp+38h] [rbp-48h] BYREF
  __int64 v21; // [rsp+48h] [rbp-38h]
  __int64 v22; // [rsp+50h] [rbp-30h]

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v1 )
  {
    if ( !*(_DWORD *)(a1 + 20) )
      return;
    v2 = *(unsigned int *)(a1 + 24);
    if ( (unsigned int)v2 <= 0x40 )
      goto LABEL_4;
    sub_1D64100(a1);
    if ( *(_DWORD *)(a1 + 24) )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_DWORD *)(a1 + 24) = 0;
      goto LABEL_35;
    }
    goto LABEL_40;
  }
  v11 = 4 * v1;
  v2 = *(unsigned int *)(a1 + 24);
  if ( (unsigned int)(4 * v1) < 0x40 )
    v11 = 64;
  if ( (unsigned int)v2 > v11 )
  {
    v12 = 64;
    sub_1D64100(a1);
    v13 = v1 - 1;
    if ( v13 )
    {
      _BitScanReverse(&v14, v13);
      v12 = 1 << (33 - (v14 ^ 0x1F));
      if ( v12 < 64 )
        v12 = 64;
    }
    if ( *(_DWORD *)(a1 + 24) != v12 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 8));
      v15 = sub_1454B60(4 * v12 / 3u + 1);
      *(_DWORD *)(a1 + 24) = v15;
      if ( !v15 )
      {
LABEL_35:
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        return;
      }
      *(_QWORD *)(a1 + 8) = sub_22077B0((unsigned __int64)v15 << 6);
    }
LABEL_40:
    sub_1D671E0(a1);
    return;
  }
LABEL_4:
  v3 = *(_QWORD **)(a1 + 8);
  v16[0] = 2;
  v16[1] = 0;
  v4 = &v3[8 * v2];
  v17 = -8;
  v18 = 0;
  v20[0] = 2;
  v20[1] = 0;
  v21 = -16;
  v19 = &unk_49F9E38;
  v22 = 0;
  if ( v3 == v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_23;
  }
  v5 = -16;
  for ( i = -8; ; i = v17 )
  {
    v8 = v3[3];
    if ( v8 != i )
    {
      if ( v8 != v5 )
      {
        v7 = v3[7];
        if ( v7 == -8 || v7 == 0 || v7 == -16 )
        {
          v5 = v3[3];
        }
        else
        {
          sub_1649B30(v3 + 5);
          v5 = v3[3];
          if ( v5 == v17 )
          {
LABEL_9:
            v3[4] = v18;
            v5 = v21;
            goto LABEL_10;
          }
        }
      }
      if ( v5 != -8 && v5 != 0 && v5 != -16 )
        sub_1649B30(v3 + 1);
      v9 = v17;
      v10 = v17 == 0;
      v3[3] = v17;
      if ( v9 != -8 && !v10 && v9 != -16 )
        sub_1649AC0(v3 + 1, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_9;
    }
LABEL_10:
    v3 += 8;
    if ( v4 == v3 )
      break;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v19 = &unk_49EE2B0;
  if ( v5 != 0 && v5 != -16 && v5 != -8 )
    sub_1649B30(v20);
LABEL_23:
  if ( v17 != 0 && v17 != -8 && v17 != -16 )
    sub_1649B30(v16);
}

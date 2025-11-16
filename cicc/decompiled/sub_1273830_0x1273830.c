// Function: sub_1273830
// Address: 0x1273830
//
void __fastcall sub_1273830(_QWORD *a1, const __m128i *a2, __int64 a3)
{
  __int64 v5; // rbx
  signed __int32 v6; // eax
  __int64 v7; // rdi
  signed __int32 v8; // eax
  signed __int32 v9; // ecx
  signed __int32 v10; // ecx
  signed __int32 v11; // ecx
  signed __int32 v12; // ecx
  const char *v13; // rdx
  _DWORD v14[9]; // [rsp+Ch] [rbp-24h] BYREF

  v5 = *(_QWORD *)(a3 + 328);
  if ( !v5 )
  {
    if ( (*(_BYTE *)(a3 + 199) & 0x20) == 0 )
      return;
    sub_1273260(a1, a2, (__int64)"cluster_dim_x", 0);
    sub_1273260(a1, a2, (__int64)"cluster_dim_y", 0);
    v12 = 0;
    v13 = "cluster_dim_z";
LABEL_22:
    sub_1273260(a1, a2, (__int64)v13, v12);
    return;
  }
  if ( *(_QWORD *)v5 )
  {
    v6 = sub_620FA0(*(_QWORD *)v5, v14);
    if ( v14[0] != 1 && v6 > 0 )
      sub_1273260(a1, a2, (__int64)"maxntidx", v6);
  }
  v7 = *(_QWORD *)(v5 + 8);
  if ( v7 )
  {
    v8 = sub_620FA0(v7, v14);
    if ( v14[0] != 1 && v8 > 0 )
      sub_1273260(a1, a2, (__int64)"minctasm", v8);
  }
  v9 = *(_DWORD *)(v5 + 32);
  if ( v9 > 0 )
  {
    sub_1273260(a1, a2, (__int64)"maxnreg", v9);
    v10 = *(_DWORD *)(v5 + 36);
    if ( v10 <= 0 )
    {
LABEL_12:
      if ( (*(_BYTE *)(a3 + 199) & 0x20) == 0 )
        goto LABEL_13;
      goto LABEL_21;
    }
  }
  else
  {
    v10 = *(_DWORD *)(v5 + 36);
    if ( v10 <= 0 )
      goto LABEL_12;
  }
  sub_1273260(a1, a2, (__int64)"local_maxnreg", v10);
  if ( (*(_BYTE *)(a3 + 199) & 0x20) != 0 )
  {
LABEL_21:
    sub_1273260(a1, a2, (__int64)"cluster_dim_x", 0);
    sub_1273260(a1, a2, (__int64)"cluster_dim_y", 0);
    sub_1273260(a1, a2, (__int64)"cluster_dim_z", 0);
    v12 = *(_DWORD *)(v5 + 16);
    v13 = "cluster_max_blocks";
    if ( v12 <= 0 )
      return;
    goto LABEL_22;
  }
LABEL_13:
  v11 = *(_DWORD *)(v5 + 20);
  if ( v11 > 0 )
  {
    sub_1273260(a1, a2, (__int64)"cluster_dim_x", v11);
    sub_1273260(a1, a2, (__int64)"cluster_dim_y", *(_DWORD *)(v5 + 24));
    sub_1273260(a1, a2, (__int64)"cluster_dim_z", *(_DWORD *)(v5 + 28));
  }
  v12 = *(_DWORD *)(v5 + 16);
  v13 = "cluster_max_blocks";
  if ( v12 > 0 )
    goto LABEL_22;
}

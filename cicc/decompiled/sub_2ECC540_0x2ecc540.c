// Function: sub_2ECC540
// Address: 0x2ecc540
//
__int64 __fastcall sub_2ECC540(_QWORD *a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // r12
  __int64 v4; // rcx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 (__fastcall *v8)(__int64); // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  char *v12; // rsi
  __int64 v13; // [rsp+8h] [rbp-58h] BYREF
  const char *v14; // [rsp+10h] [rbp-50h] BYREF
  const char *v15; // [rsp+18h] [rbp-48h]
  char v16; // [rsp+30h] [rbp-30h]
  char v17; // [rsp+31h] [rbp-2Fh]

  v1 = sub_22077B0(0x690u);
  v3 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = a1;
    *(_QWORD *)(v1 + 16) = 0;
    *(_QWORD *)(v1 + 24) = 0;
    *(_DWORD *)(v1 + 32) = 0;
    *(_WORD *)(v1 + 36) = 0;
    *(_QWORD *)(v1 + 56) = v1 + 72;
    *(_QWORD *)(v1 + 64) = 0x1000000000LL;
    *(_QWORD *)(v1 + 40) = 0;
    v17 = 1;
    *(_QWORD *)v1 = &unk_4A29FE0;
    *(_DWORD *)(v1 + 48) = 0;
    *(_BYTE *)(v1 + 52) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    v14 = "TopQ";
    v16 = 3;
    sub_2EC8790(v1 + 144, 1, (char *)&v14, v2);
    v14 = "BotQ";
    v17 = 1;
    v16 = 3;
    sub_2EC8790(v3 + 864, 2, (char *)&v14, v4);
    *(_BYTE *)(v3 + 1584) = 0;
    *(_QWORD *)(v3 + 1588) = 0;
    *(_QWORD *)(v3 + 1600) = 0;
    *(_QWORD *)(v3 + 1608) = 0;
    *(_DWORD *)(v3 + 1616) = 0;
    *(_WORD *)(v3 + 1620) = 0;
    *(_QWORD *)(v3 + 1624) = 0;
    *(_BYTE *)(v3 + 1632) = 0;
    *(_QWORD *)(v3 + 1636) = 0;
    *(_QWORD *)(v3 + 1672) = 0;
    *(_QWORD *)(v3 + 1648) = 0;
    *(_QWORD *)(v3 + 1656) = 0;
    *(_DWORD *)(v3 + 1664) = 0;
    *(_WORD *)(v3 + 1668) = 0;
    v5 = sub_22077B0(0xDD8u);
    if ( v5 )
    {
LABEL_3:
      sub_2F91670(v5, a1[1], a1[2], 1);
      *(_QWORD *)(v5 + 3472) = v3;
      *(_QWORD *)(v5 + 3480) = 0;
      *(_QWORD *)(v5 + 3488) = 0;
      *(_QWORD *)v5 = &unk_4A29DA8;
      v6 = a1[5];
      *(_QWORD *)(v5 + 3496) = 0;
      *(_QWORD *)(v5 + 3456) = v6;
      v7 = a1[6];
      *(_QWORD *)(v5 + 3504) = 0;
      *(_QWORD *)(v5 + 3464) = v7;
      *(_QWORD *)(v5 + 3512) = 0;
      *(_QWORD *)(v5 + 3520) = 0;
      *(_QWORD *)(v5 + 3528) = 0;
      *(_DWORD *)(v5 + 3536) = 0;
      goto LABEL_4;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v3 + 16LL))(v3);
  }
  else
  {
    v5 = sub_22077B0(0xDD8u);
    if ( v5 )
      goto LABEL_3;
  }
LABEL_4:
  v8 = *(__int64 (__fastcall **)(__int64))(**(_QWORD **)(a1[1] + 16LL) + 480LL);
  if ( v8 == sub_2EC09A0 )
    return v5;
  v8((__int64)&v14);
  v10 = (unsigned __int64)v14;
  if ( v15 != v14 )
  {
    sub_2F06BD0(&v13, v14, (v15 - v14) >> 3, 0);
    v11 = v13;
    if ( !v13 )
    {
LABEL_12:
      v10 = (unsigned __int64)v14;
      goto LABEL_13;
    }
    v12 = *(char **)(v5 + 3488);
    if ( v12 == *(char **)(v5 + 3496) )
    {
      sub_2ECB480((unsigned __int64 *)(v5 + 3480), v12, &v13);
      v11 = v13;
      if ( !v13 )
        goto LABEL_12;
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v13;
        *(_QWORD *)(v5 + 3488) += 8LL;
        goto LABEL_12;
      }
      *(_QWORD *)(v5 + 3488) = 8;
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 16LL))(v11);
    v10 = (unsigned __int64)v14;
  }
LABEL_13:
  if ( !v10 )
    return v5;
  j_j___libc_free_0(v10);
  return v5;
}

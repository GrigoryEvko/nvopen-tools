// Function: sub_2210BD0
// Address: 0x2210bd0
//
__int64 __fastcall sub_2210BD0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rbp
  __int64 v8; // rbp
  __int64 v9; // rbp
  __int64 v10; // rbp
  __int64 v11; // rbp
  __int64 v12; // rbp
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 result; // rax
  __int64 v16; // [rsp+8h] [rbp-50h]
  __int64 v17; // [rsp+8h] [rbp-50h]
  __int64 v18; // [rsp+8h] [rbp-50h]
  __int64 v19; // [rsp+8h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-50h]
  __int64 v22; // [rsp+8h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-50h]
  __int64 v24; // [rsp+8h] [rbp-50h]
  __int64 v25; // [rsp+10h] [rbp-48h]
  __int64 v26; // [rsp+18h] [rbp-40h]
  __int64 v27; // [rsp+18h] [rbp-40h]
  __int64 v28; // [rsp+18h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-40h]
  __int64 v30; // [rsp+18h] [rbp-40h]
  __int64 v31; // [rsp+18h] [rbp-40h]
  __int64 v32; // [rsp+18h] [rbp-40h]

  v2 = *a2;
  v3 = a2[1];
  v4 = a2[2];
  dword_4FD6778 = 1;
  qword_4FD6780 = v2;
  qword_4FD6770 = (__int64)off_4A06B78;
  sub_220E920((__int64)&qword_4FD6770, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6778, 1u);
  else
    ++dword_4FD6778;
  v16 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v16 + 8 * sub_22091A0(&qword_4FD69B8)) = &qword_4FD6770;
  dword_4FD6798 = 1;
  qword_4FD6790 = (__int64)off_4A06B40;
  qword_4FD67A0 = sub_2208E60();
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6798, 1u);
  else
    ++dword_4FD6798;
  v17 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v17 + 8 * sub_22091A0(&qword_4FD6980)) = &qword_4FD6790;
  dword_4FD6738 = 1;
  qword_4FD6740 = v3;
  qword_4FD6730 = (__int64)off_4A06D40;
  sub_220CCA0((__int64)&qword_4FD6730, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6738, 1u);
  else
    ++dword_4FD6738;
  v18 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v18 + 8 * sub_22091A0(&qword_4FD69D8)) = &qword_4FD6730;
  dword_4FD6758 = 1;
  qword_4FD6760 = v4;
  qword_4FD6750 = (__int64)off_4A06CD8;
  sub_220C790((__int64)&qword_4FD6750, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6758, 1u);
  else
    ++dword_4FD6758;
  v19 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v19 + 8 * sub_22091A0(&qword_4FD69D0)) = &qword_4FD6750;
  dword_4FD6728 = 1;
  qword_4FD6720 = (__int64)off_4A06E38;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6728, 1u);
  else
    dword_4FD6728 = 2;
  v20 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v20 + 8 * sub_22091A0(&qword_4FD69C8)) = &qword_4FD6720;
  dword_4FD6718 = 1;
  qword_4FD6710 = (__int64)off_4A06E68;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6718, 1u);
  else
    dword_4FD6718 = 2;
  v21 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v21 + 8 * sub_22091A0(&qword_4FD69C0)) = &qword_4FD6710;
  dword_4FD6708 = 1;
  qword_4FD6700 = (__int64)off_4A06EC0;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6708, 1u);
  else
    dword_4FD6708 = 2;
  v22 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v22 + 8 * sub_22091A0(&qword_4FD6990)) = &qword_4FD6700;
  sub_222F740(&unk_4FD66E0, 1);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD66E8, 1u);
  else
    ++dword_4FD66E8;
  v23 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v23 + 8 * sub_22091A0(&qword_4FD6988)) = &unk_4FD66E0;
  v5 = a2[4];
  dword_4FD6698 = 1;
  v24 = v5;
  v25 = a2[3];
  v6 = a2[5];
  qword_4FD66A0 = v25;
  qword_4FD6690 = (__int64)off_4A07980;
  sub_220EC20((__int64)&qword_4FD6690, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6698, 1u);
  else
    ++dword_4FD6698;
  v26 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v26 + 8 * sub_22091A0(&qword_4FD6A78)) = &qword_4FD6690;
  dword_4FD66B8 = 1;
  qword_4FD66B0 = (__int64)off_4A078C8;
  qword_4FD66C0 = sub_2208E60();
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD66B8, 1u);
  else
    ++dword_4FD66B8;
  v27 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v27 + 8 * sub_22091A0(&qword_4FD6A40)) = &qword_4FD66B0;
  dword_4FD6658 = 1;
  qword_4FD6650 = (__int64)off_4A07B48;
  qword_4FD6660 = v24;
  sub_220D930((__int64)&qword_4FD6650, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6658, 1u);
  else
    ++dword_4FD6658;
  v28 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v28 + 8 * sub_22091A0(&qword_4FD6A98)) = &qword_4FD6650;
  dword_4FD6678 = 1;
  qword_4FD6680 = v6;
  qword_4FD6670 = (__int64)off_4A07AE0;
  sub_220D3C0((__int64)&qword_4FD6670, 0);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6678, 1u);
  else
    ++dword_4FD6678;
  v29 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v29 + 8 * sub_22091A0(&qword_4FD6A90)) = &qword_4FD6670;
  dword_4FD6648 = 1;
  qword_4FD6640 = (__int64)off_4A07C40;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6648, 1u);
  else
    dword_4FD6648 = 2;
  v30 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v30 + 8 * sub_22091A0(&qword_4FD6A88)) = &qword_4FD6640;
  dword_4FD6638 = 1;
  qword_4FD6630 = (__int64)off_4A07C70;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6638, 1u);
  else
    dword_4FD6638 = 2;
  v31 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v31 + 8 * sub_22091A0(&qword_4FD6A80)) = &qword_4FD6630;
  dword_4FD6628 = 1;
  qword_4FD6620 = (__int64)off_4A07CC8;
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6628, 1u);
  else
    dword_4FD6628 = 2;
  v32 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v32 + 8 * sub_22091A0(&qword_4FD6A50)) = &qword_4FD6620;
  sub_22430D0(&unk_4FD6600, 1);
  if ( &_pthread_key_create )
    _InterlockedAdd(&dword_4FD6608, 1u);
  else
    ++dword_4FD6608;
  v7 = *(_QWORD *)(a1 + 8);
  *(_QWORD *)(v7 + 8 * sub_22091A0(&qword_4FD6A48)) = &unk_4FD6600;
  v8 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v8 + 8 * sub_22091A0(&qword_4FD69B8)) = v2;
  v9 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v9 + 8 * sub_22091A0(&qword_4FD69D8)) = v3;
  v10 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v10 + 8 * sub_22091A0(&qword_4FD69D0)) = v4;
  v11 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v11 + 8 * sub_22091A0(&qword_4FD6A78)) = v25;
  v12 = *(_QWORD *)(a1 + 24);
  v13 = sub_22091A0(&qword_4FD6A98);
  v14 = *(_QWORD *)(a1 + 24);
  *(_QWORD *)(v12 + 8 * v13) = v24;
  result = sub_22091A0(&qword_4FD6A90);
  *(_QWORD *)(v14 + 8 * result) = v6;
  return result;
}

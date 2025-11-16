// Function: sub_258E5F0
// Address: 0x258e5f0
//
char __fastcall sub_258E5F0(__int64 a1, unsigned __int64 *a2, __int64 a3)
{
  unsigned __int64 *v3; // r13
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rsi
  unsigned __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r15
  __int64 (__fastcall *v11)(__int64); // rdx
  bool (__fastcall *v12)(__int64); // rax
  int v13; // edx
  __int64 v15; // [rsp-10h] [rbp-60h]
  unsigned __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  __int64 v17; // [rsp+18h] [rbp-38h]

  v3 = &a2[a3];
  v4 = a2;
  if ( v3 == a2 )
    return 1;
  while ( 1 )
  {
    v5 = *v4;
    if ( **(_BYTE **)a1 == 3 )
    {
      sub_250D230(&v16, v5, 2, 0);
    }
    else
    {
      v17 = 0;
      v16 = v5 & 0xFFFFFFFFFFFFFFFCLL;
      nullsub_1518();
    }
    v6 = v16;
    v7 = sub_258DCE0(*(_QWORD *)(a1 + 16), v16, v17, *(_QWORD *)(a1 + 24), 0, 0, 1);
    if ( !v7 )
      return 0;
    v8 = (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)v7 + 48LL))(v7, v6, v15);
    v9 = sub_25538A0(*(_QWORD *)(a1 + 40), v8);
    sub_250C0C0(*(int **)(a1 + 32), v9);
    v10 = *(_QWORD *)(a1 + 40);
    v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 24LL);
    if ( v11 != sub_2539310 )
    {
      if ( (unsigned __int8)v11(*(_QWORD *)(a1 + 40)) )
      {
        v10 = *(_QWORD *)(a1 + 40);
        v12 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL);
        goto LABEL_8;
      }
      goto LABEL_13;
    }
    v12 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL);
    if ( v12 == sub_2505DB0 )
    {
      v13 = *(_DWORD *)(v10 + 20);
      if ( !v13 )
        goto LABEL_8;
    }
    else
    {
      if ( !v12(v10) )
        break;
      v13 = *(_DWORD *)(v10 + 20);
    }
    if ( v13 == *(_DWORD *)(v10 + 16) && *(_BYTE *)(v10 + 81) == *(_BYTE *)(v10 + 80) )
      break;
LABEL_13:
    if ( v3 == ++v4 )
      return 1;
  }
  v10 = *(_QWORD *)(a1 + 40);
  v12 = *(bool (__fastcall **)(__int64))(*(_QWORD *)v10 + 16LL);
LABEL_8:
  if ( v12 == sub_2505DB0 )
    return *(_DWORD *)(v10 + 20) != 0;
  else
    return v12(v10);
}

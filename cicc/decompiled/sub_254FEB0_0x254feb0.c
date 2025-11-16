// Function: sub_254FEB0
// Address: 0x254feb0
//
__int64 __fastcall sub_254FEB0(__int64 a1)
{
  __int64 v1; // rsi
  __int64 v3; // r14
  __int64 v4; // rsi
  unsigned __int64 v5; // rdi
  __int64 **v7; // r12
  __int64 *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx

  v1 = *(unsigned int *)(a1 + 256);
  *(_QWORD *)a1 = off_4A190D8;
  *(_QWORD *)(a1 + 88) = &unk_4A19168;
  sub_C7D6A0(*(_QWORD *)(a1 + 240), 16 * v1, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 208), 8LL * *(unsigned int *)(a1 + 224), 8);
  *(_QWORD *)a1 = &unk_4A18FE8;
  *(_QWORD *)(a1 + 88) = &unk_4A19078;
  if ( !*(_DWORD *)(a1 + 192) )
  {
    v3 = *(_QWORD *)(a1 + 176);
    v4 = 0;
    goto LABEL_3;
  }
  if ( !byte_4FEF2C0[0] && (unsigned int)sub_2207590((__int64)byte_4FEF2C0) )
  {
    unk_4FEF2E0 = -4096;
    unk_4FEF2E8 = -4096;
    unk_4FEF2F0 = 0;
    unk_4FEF2F8 = 0;
    sub_2207640((__int64)byte_4FEF2C0);
  }
  if ( !byte_4FEF280[0] && (unsigned int)sub_2207590((__int64)byte_4FEF280) )
  {
    unk_4FEF2A0 = -8192;
    unk_4FEF2A8 = -8192;
    unk_4FEF2B0 = 0;
    unk_4FEF2B8 = 0;
    sub_2207640((__int64)byte_4FEF280);
  }
  v7 = *(__int64 ***)(a1 + 176);
  v4 = *(unsigned int *)(a1 + 192);
  v3 = (__int64)&v7[v4];
  if ( v7 != &v7[v4] )
  {
    while ( 1 )
    {
      v8 = *v7;
      v9 = **v7;
      v10 = (*v7)[1];
      if ( unk_4FEF2E0 != v9 || unk_4FEF2E8 != v10 )
        goto LABEL_11;
      if ( !sub_254C7C0((__int64 *)v8[2], unk_4FEF2F0) )
        break;
LABEL_14:
      if ( (__int64 **)v3 == ++v7 )
      {
        v3 = *(_QWORD *)(a1 + 176);
        v4 = *(unsigned int *)(a1 + 192);
        goto LABEL_3;
      }
    }
    v8 = *v7;
    v9 = **v7;
    v10 = (*v7)[1];
LABEL_11:
    if ( unk_4FEF2A8 == v10 && unk_4FEF2A0 == v9 )
      sub_254C7C0((__int64 *)v8[2], unk_4FEF2B0);
    goto LABEL_14;
  }
LABEL_3:
  sub_C7D6A0(v3, v4 * 8, 8);
  v5 = *(_QWORD *)(a1 + 104);
  if ( v5 != a1 + 120 )
    _libc_free(v5);
  *(_QWORD *)a1 = &unk_4A16C00;
  return sub_254FD20(a1 + 8);
}

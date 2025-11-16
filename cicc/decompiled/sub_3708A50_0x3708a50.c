// Function: sub_3708A50
// Address: 0x3708a50
//
unsigned __int64 *__fastcall sub_3708A50(unsigned __int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  int v5; // r8d
  __int16 v6; // ax
  __int16 v7; // dx
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r12
  int v15; // r8d
  unsigned __int16 v16; // ax
  unsigned __int16 v17; // dx
  __int64 v18; // r12
  int v19; // r8d
  unsigned __int64 v20; // r12
  unsigned __int64 v21; // rax
  int v22; // r8d
  unsigned __int64 v23; // r12
  unsigned __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // r12
  unsigned __int64 v27; // [rsp+8h] [rbp-68h] BYREF
  unsigned __int64 *v28; // [rsp+10h] [rbp-60h] BYREF
  __int64 v29; // [rsp+18h] [rbp-58h]
  char v30; // [rsp+30h] [rbp-40h]
  char v31; // [rsp+31h] [rbp-3Fh]

  v28 = 0;
  v29 = 0;
  sub_1254950(&v27, a2, (__int64)&v28, 2u);
  v4 = v27 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
LABEL_6:
    *a1 = v4 | 1;
  }
  else
  {
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
    v6 = *(_WORD *)v28;
    v7 = __ROL2__(*(_WORD *)v28, 8);
    if ( v5 != 1 )
      v6 = v7;
    if ( v6 >= 0 )
    {
      v9 = (unsigned __int16)v6;
      if ( *(_DWORD *)(a3 + 8) > 0x40u )
      {
LABEL_9:
        if ( *(_QWORD *)a3 )
          j_j___libc_free_0_0(*(_QWORD *)a3);
      }
LABEL_11:
      *(_QWORD *)a3 = v9;
      *(_DWORD *)(a3 + 8) = 16;
      *(_BYTE *)(a3 + 12) = 1;
      *a1 = 1;
    }
    else
    {
      switch ( v6 )
      {
        case -32768:
          v28 = 0;
          v29 = 0;
          sub_1254950(&v27, a2, (__int64)&v28, 1u);
          v4 = v27 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
          v14 = *(unsigned __int8 *)v28;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v14;
          *(_DWORD *)(a3 + 8) = 8;
          *(_BYTE *)(a3 + 12) = 0;
          *a1 = 1;
          break;
        case -32767:
          v28 = 0;
          v29 = 0;
          sub_1254950(&v27, a2, (__int64)&v28, 2u);
          v4 = v27 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v15 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
          v16 = *(_WORD *)v28;
          v17 = __ROL2__(*(_WORD *)v28, 8);
          if ( v15 != 1 )
            v16 = v17;
          v18 = v16;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v18;
          *(_DWORD *)(a3 + 8) = 16;
          *(_BYTE *)(a3 + 12) = 0;
          *a1 = 1;
          break;
        case -32766:
          sub_3708870((unsigned __int64 *)&v28, a2, &v27);
          v4 = (unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v9 = (unsigned __int16)v27;
          if ( *(_DWORD *)(a3 + 8) > 0x40u )
            goto LABEL_9;
          goto LABEL_11;
        case -32765:
          sub_3708910((unsigned __int64 *)&v28, a2, (unsigned __int32 *)&v27);
          v4 = (unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v26 = (unsigned int)v27;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v26;
          *(_DWORD *)(a3 + 8) = 32;
          *(_BYTE *)(a3 + 12) = 0;
          *a1 = 1;
          break;
        case -32764:
          sub_37089B0((unsigned __int64 *)&v28, a2, (unsigned __int32 *)&v27);
          v4 = (unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL;
          if ( ((unsigned __int64)v28 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v25 = (unsigned int)v27;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v25;
          *(_DWORD *)(a3 + 8) = 32;
          *(_BYTE *)(a3 + 12) = 1;
          *a1 = 1;
          break;
        case -32759:
          v28 = 0;
          v29 = 0;
          sub_1254950(&v27, a2, (__int64)&v28, 8u);
          v4 = v27 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v22 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
          v23 = *v28;
          v24 = _byteswap_uint64(*v28);
          if ( v22 != 1 )
            v23 = v24;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v23;
          *(_DWORD *)(a3 + 8) = 64;
          *(_BYTE *)(a3 + 12) = 0;
          *a1 = 1;
          break;
        case -32758:
          v28 = 0;
          v29 = 0;
          sub_1254950(&v27, a2, (__int64)&v28, 8u);
          v4 = v27 & 0xFFFFFFFFFFFFFFFELL;
          if ( (v27 & 0xFFFFFFFFFFFFFFFELL) != 0 )
            goto LABEL_6;
          v19 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 24) + 16LL))(*(_QWORD *)(a2 + 24));
          v20 = *v28;
          v21 = _byteswap_uint64(*v28);
          if ( v19 != 1 )
            v20 = v21;
          if ( *(_DWORD *)(a3 + 8) > 0x40u && *(_QWORD *)a3 )
            j_j___libc_free_0_0(*(_QWORD *)a3);
          *(_QWORD *)a3 = v20;
          *(_DWORD *)(a3 + 8) = 64;
          *(_BYTE *)(a3 + 12) = 1;
          *a1 = 1;
          break;
        default:
          v10 = sub_37F93E0();
          v31 = 1;
          v11 = v10;
          v30 = 3;
          v28 = (unsigned __int64 *)"Buffer contains invalid APSInt type";
          v12 = sub_22077B0(0x40u);
          v13 = v12;
          if ( v12 )
          {
            sub_C63E60(v12, 4, v11, (__int64)&v28);
            *(_QWORD *)v13 = &unk_4A3C5B0;
          }
          *a1 = v13 | 1;
          break;
      }
    }
  }
  return a1;
}

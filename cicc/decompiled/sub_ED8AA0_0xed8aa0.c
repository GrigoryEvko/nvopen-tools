// Function: sub_ED8AA0
// Address: 0xed8aa0
//
__int64 __fastcall sub_ED8AA0(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // rax
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r12
  char v10; // al
  __int64 v11; // rax
  int v12; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v13[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (unsigned __int8)sub_ED8700(*a2) )
  {
    v6 = *a2;
    *a2 = 0;
    v7 = *a3;
    *a3 = 0;
    v8 = sub_22077B0(496);
    v9 = v8;
    if ( v8 )
    {
      *(_DWORD *)(v8 + 8) = 0;
      *(_QWORD *)(v8 + 16) = v8 + 32;
      *(_QWORD *)(v8 + 56) = v8 + 72;
      *(_QWORD *)(v8 + 64) = 0x100000000LL;
      *(_QWORD *)(v8 + 24) = 0;
      *(_BYTE *)(v8 + 32) = 0;
      *(_QWORD *)v8 = &unk_49E4DE8;
      *(_QWORD *)(v8 + 168) = v8 + 184;
      *(_QWORD *)(v8 + 48) = 0;
      *(_QWORD *)(v8 + 104) = 0;
      *(_QWORD *)(v8 + 112) = v6;
      *(_QWORD *)(v8 + 120) = v7;
      *(_QWORD *)(v8 + 128) = 0;
      *(_QWORD *)(v8 + 136) = 0;
      *(_QWORD *)(v8 + 144) = 0;
      *(_QWORD *)(v8 + 152) = 0;
      *(_QWORD *)(v8 + 160) = 2;
      *(_QWORD *)(v8 + 176) = 0x1C00000000LL;
      *(_QWORD *)(v8 + 408) = 0;
      *(_QWORD *)(v8 + 416) = 0;
      *(_QWORD *)(v8 + 424) = 0;
      *(_QWORD *)(v8 + 432) = 0;
      *(_QWORD *)(v8 + 440) = 0;
      *(_DWORD *)(v8 + 448) = 0;
      *(_QWORD *)(v8 + 456) = 0;
      *(_QWORD *)(v8 + 464) = 0;
      *(_QWORD *)(v8 + 472) = 0;
      *(_QWORD *)(v8 + 480) = 0;
      *(_DWORD *)(v8 + 488) = 0;
    }
    else
    {
      if ( v7 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v7 + 8LL))(v7);
      if ( v6 )
        (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 8LL))(v6);
    }
    (*(void (__fastcall **)(__int64 *, __int64))(*(_QWORD *)v9 + 16LL))(v13, v9);
    if ( (v13[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
    {
      *(_QWORD *)a1 = v13[0] & 0xFFFFFFFFFFFFFFFELL;
      v11 = *(_QWORD *)v9;
      *(_BYTE *)(a1 + 8) |= 3u;
      (*(void (__fastcall **)(__int64))(v11 + 8))(v9);
    }
    else
    {
      v10 = *(_BYTE *)(a1 + 8);
      *(_QWORD *)a1 = v9;
      *(_BYTE *)(a1 + 8) = v10 & 0xFC | 2;
    }
  }
  else
  {
    v12 = 3;
    sub_ED8A30(v13, &v12);
    v4 = v13[0];
    *(_BYTE *)(a1 + 8) |= 3u;
    *(_QWORD *)a1 = v4 & 0xFFFFFFFFFFFFFFFELL;
  }
  return a1;
}

// Function: sub_1D04DC0
// Address: 0x1d04dc0
//
__int64 __fastcall sub_1D04DC0(__int64 a1)
{
  __int64 v2; // r12
  __int64 *v3; // r15
  __int64 v4; // rax
  __int64 (*v5)(); // rdx
  __int64 (*v6)(); // rax
  _QWORD *v7; // r14
  __int64 v8; // r15
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _BYTE *v12; // rdx
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r8
  __int64 *v16; // r12
  __int64 v17; // rsi
  __int64 (*v18)(); // r10
  _DWORD *v19; // r15
  int v20; // eax
  __int64 v21; // rax
  __int64 v22; // r12
  __int64 v23; // r13
  bool v24; // zf
  __int64 v25; // rax
  __int64 v27; // r8
  unsigned __int64 v28; // rax
  _BYTE *v29; // rdx
  _BYTE *v30; // rdi
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 (*v33)(); // rax
  __int64 v34; // rax
  __int64 *v35; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = *(__int64 **)(*(_QWORD *)(a1 + 256) + 16LL);
  v4 = *v3;
  v5 = *(__int64 (**)())(*v3 + 40);
  if ( v5 != sub_1D00B00 )
  {
    v2 = ((__int64 (__fastcall *)(_QWORD))v5)(*(_QWORD *)(*(_QWORD *)(a1 + 256) + 16LL));
    v4 = *v3;
  }
  v6 = *(__int64 (**)())(v4 + 112);
  v7 = 0;
  if ( v6 != sub_1D00B10 )
    v7 = (_QWORD *)((__int64 (__fastcall *)(__int64 *))v6)(v3);
  v8 = *(_QWORD *)(a1 + 320);
  v9 = sub_22077B0(176);
  if ( v9 )
  {
    v10 = *(_QWORD *)(a1 + 256);
    *(_QWORD *)(v9 + 64) = v2;
    *(_DWORD *)(v9 + 8) = 0;
    *(_BYTE *)(v9 + 12) = 0;
    *(_QWORD *)v9 = &off_49F9520;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 24) = 0;
    *(_QWORD *)(v9 + 32) = 0;
    *(_DWORD *)(v9 + 40) = 0;
    *(_WORD *)(v9 + 44) = 1;
    *(_QWORD *)(v9 + 56) = v10;
    *(_QWORD *)(v9 + 72) = v7;
    *(_QWORD *)(v9 + 80) = v8;
    *(_QWORD *)(v9 + 88) = 0;
    *(_QWORD *)(v9 + 96) = 0;
    *(_QWORD *)(v9 + 104) = 0;
    *(_QWORD *)(v9 + 112) = 0;
    *(_QWORD *)(v9 + 120) = 0;
    *(_QWORD *)(v9 + 128) = 0;
    *(_QWORD *)(v9 + 136) = 0;
    *(_QWORD *)(v9 + 144) = 0;
    *(_QWORD *)(v9 + 152) = 0;
    *(_QWORD *)(v9 + 160) = 0;
    v11 = (unsigned int)((__int64)(v7[33] - v7[32]) >> 3);
    if ( (_DWORD)v11 )
    {
      sub_C17A60(v9 + 144, v11);
      v27 = *(_QWORD *)(v9 + 120);
      v28 = (*(_QWORD *)(v9 + 128) - v27) >> 2;
      if ( (unsigned int)v11 > v28 )
      {
        sub_C17A60(v9 + 120, (unsigned int)v11 - v28);
        v29 = *(_BYTE **)(v9 + 152);
        v30 = *(_BYTE **)(v9 + 144);
      }
      else
      {
        v29 = *(_BYTE **)(v9 + 152);
        v30 = *(_BYTE **)(v9 + 144);
        if ( (unsigned int)v11 < v28 )
        {
          v31 = v27 + 4LL * (unsigned int)v11;
          if ( *(_QWORD *)(v9 + 128) != v31 )
            *(_QWORD *)(v9 + 128) = v31;
        }
      }
      if ( v29 != v30 )
        memset(v30, 0, v29 - v30);
    }
    v12 = *(_BYTE **)(v9 + 128);
    v13 = *(_BYTE **)(v9 + 120);
    if ( v12 != v13 )
      memset(v13, 0, v12 - v13);
    v14 = *(_QWORD *)(v9 + 72);
    v15 = *(__int64 **)(v14 + 264);
    v16 = *(__int64 **)(v14 + 256);
    if ( v15 != v16 )
    {
      do
      {
        while ( 1 )
        {
          v17 = *v16;
          v18 = *(__int64 (**)())(*v7 + 168LL);
          v19 = (_DWORD *)(*(_QWORD *)(v9 + 144) + 4LL * *(unsigned __int16 *)(*(_QWORD *)*v16 + 24LL));
          if ( v18 != sub_1D00B20 )
            break;
          ++v16;
          *v19 = 0;
          if ( v15 == v16 )
            goto LABEL_14;
        }
        v35 = v15;
        ++v16;
        v20 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, __int64 (*)()))v18)(
                v7,
                v17,
                *(_QWORD *)(v9 + 56),
                sub_1D00B20);
        v15 = v35;
        *v19 = v20;
      }
      while ( v35 != v16 );
    }
LABEL_14:
    *(_QWORD *)(v9 + 168) = v9;
    *(_QWORD *)v9 = off_49F9780;
  }
  v21 = sub_22077B0(944);
  v22 = v21;
  if ( v21 )
  {
    v23 = *(_QWORD *)(a1 + 256);
    sub_1D0DA30(v21, v23);
    *(_BYTE *)(v22 + 664) = 1;
    *(_QWORD *)v22 = off_49F94A0;
    *(_QWORD *)(v22 + 744) = v22 + 760;
    *(_QWORD *)(v22 + 672) = v9;
    *(_QWORD *)(v22 + 680) = 0;
    *(_QWORD *)(v22 + 688) = 0;
    *(_QWORD *)(v22 + 696) = 0;
    *(_DWORD *)(v22 + 712) = 0;
    *(_QWORD *)(v22 + 728) = 0;
    *(_QWORD *)(v22 + 736) = 0;
    *(_QWORD *)(v22 + 752) = 0x400000000LL;
    *(_QWORD *)(v22 + 792) = 0;
    *(_QWORD *)(v22 + 800) = 0;
    *(_QWORD *)(v22 + 808) = 0;
    *(_DWORD *)(v22 + 816) = 0;
    sub_1F024C0(v22 + 824, v22 + 48, 0);
    v24 = byte_4FC13A0 == 0;
    *(_QWORD *)(v22 + 912) = 0;
    *(_QWORD *)(v22 + 920) = 0;
    *(_QWORD *)(v22 + 928) = 0;
    *(_DWORD *)(v22 + 936) = 0;
    if ( v24 && *(_BYTE *)(v22 + 664) )
    {
      v32 = *(_QWORD *)(v23 + 16);
      v33 = *(__int64 (**)())(*(_QWORD *)v32 + 40LL);
      if ( v33 == sub_1D00B00 )
        BUG();
      v34 = ((__int64 (__fastcall *)(__int64))v33)(v32);
      *(_QWORD *)(v22 + 704) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v34 + 752LL))(
                                 v34,
                                 v32,
                                 v22);
    }
    else
    {
      v25 = sub_22077B0(16);
      if ( v25 )
      {
        *(_DWORD *)(v25 + 8) = 0;
        *(_QWORD *)v25 = &unk_49FE598;
      }
      *(_QWORD *)(v22 + 704) = v25;
    }
  }
  *(_QWORD *)(v9 + 88) = v22;
  return v22;
}

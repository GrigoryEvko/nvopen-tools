// Function: sub_3355A10
// Address: 0x3355a10
//
__int64 __fastcall sub_3355A10(__int64 a1)
{
  __int64 v2; // r12
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 (*v5)(void); // rdx
  __int64 v6; // rax
  __int64 v7; // r15
  _QWORD *v8; // r13
  __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rsi
  _BYTE *v12; // rdx
  _BYTE *v13; // rdi
  __int64 v14; // rax
  __int64 *v15; // r8
  __int64 *i; // r12
  __int64 v17; // r15
  __int64 (*v18)(); // r10
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // r12
  __int64 v22; // r14
  _QWORD *v23; // rax
  __int64 v24; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // rax
  _BYTE *v28; // rdx
  _BYTE *v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r14
  __int64 (*v32)(); // rax
  __int64 v33; // rax
  __int64 *v34; // [rsp+8h] [rbp-38h]

  v2 = 0;
  v3 = *(__int64 **)(*(_QWORD *)(a1 + 40) + 16LL);
  v4 = *v3;
  v5 = *(__int64 (**)(void))(*v3 + 128);
  if ( v5 != sub_2DAC790 )
  {
    v2 = v5();
    v4 = *v3;
  }
  v6 = (*(__int64 (**)(void))(v4 + 200))();
  v7 = *(_QWORD *)(a1 + 808);
  v8 = (_QWORD *)v6;
  v9 = sub_22077B0(0xB0u);
  if ( v9 )
  {
    v10 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(v9 + 64) = v2;
    *(_DWORD *)(v9 + 8) = 0;
    *(_BYTE *)(v9 + 12) = 0;
    *(_QWORD *)v9 = &off_4A362C0;
    *(_QWORD *)(v9 + 16) = 0;
    *(_QWORD *)(v9 + 24) = 0;
    *(_QWORD *)(v9 + 32) = 0;
    *(_DWORD *)(v9 + 40) = 0;
    *(_WORD *)(v9 + 44) = 1;
    *(_QWORD *)(v9 + 48) = 0;
    *(_QWORD *)(v9 + 56) = v10;
    *(_QWORD *)(v9 + 72) = v8;
    *(_QWORD *)(v9 + 80) = v7;
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
    v11 = (unsigned int)((__int64)(v8[36] - v8[35]) >> 3);
    if ( (_DWORD)v11 )
    {
      sub_C17A60(v9 + 144, v11);
      v26 = *(_QWORD *)(v9 + 120);
      v27 = (*(_QWORD *)(v9 + 128) - v26) >> 2;
      if ( (unsigned int)v11 > v27 )
      {
        sub_C17A60(v9 + 120, (unsigned int)v11 - v27);
        v28 = *(_BYTE **)(v9 + 152);
        v29 = *(_BYTE **)(v9 + 144);
      }
      else
      {
        v28 = *(_BYTE **)(v9 + 152);
        v29 = *(_BYTE **)(v9 + 144);
        if ( (unsigned int)v11 < v27 )
        {
          v30 = v26 + 4LL * (unsigned int)v11;
          if ( *(_QWORD *)(v9 + 128) != v30 )
            *(_QWORD *)(v9 + 128) = v30;
        }
      }
      if ( v29 != v28 )
        memset(v29, 0, v28 - v29);
    }
    v12 = *(_BYTE **)(v9 + 128);
    v13 = *(_BYTE **)(v9 + 120);
    if ( v12 != v13 )
      memset(v13, 0, v12 - v13);
    v14 = *(_QWORD *)(v9 + 72);
    v15 = *(__int64 **)(v14 + 288);
    for ( i = *(__int64 **)(v14 + 280);
          v15 != i;
          *(_DWORD *)(*(_QWORD *)(v9 + 144) + 4LL * *(unsigned __int16 *)(*(_QWORD *)v17 + 24LL)) = v19 )
    {
      v17 = *i;
      v18 = *(__int64 (**)())(*v8 + 360LL);
      v19 = 0;
      if ( v18 != sub_2FF5280 )
      {
        v34 = v15;
        v19 = ((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD, __int64 (*)()))v18)(
                v8,
                v17,
                *(_QWORD *)(v9 + 56),
                sub_2FF5280);
        v15 = v34;
      }
      ++i;
    }
    *(_QWORD *)(v9 + 168) = v9;
    *(_QWORD *)v9 = off_4A36520;
  }
  v20 = sub_22077B0(0x5C8u);
  v21 = v20;
  if ( v20 )
  {
    v22 = *(_QWORD *)(a1 + 40);
    sub_335DCC0(v20, v22);
    *(_BYTE *)(v21 + 632) = 1;
    *(_QWORD *)v21 = off_4A36238;
    *(_QWORD *)(v21 + 680) = 0xFFFFFFFF00000000LL;
    *(_QWORD *)(v21 + 712) = v21 + 728;
    *(_QWORD *)(v21 + 720) = 0x400000000LL;
    *(_QWORD *)(v21 + 640) = v9;
    *(_QWORD *)(v21 + 648) = 0;
    *(_QWORD *)(v21 + 656) = 0;
    *(_QWORD *)(v21 + 664) = 0;
    *(_QWORD *)(v21 + 688) = 0;
    *(_QWORD *)(v21 + 696) = 0;
    *(_QWORD *)(v21 + 704) = 0;
    *(_QWORD *)(v21 + 760) = 0;
    *(_QWORD *)(v21 + 768) = 0;
    *(_QWORD *)(v21 + 776) = 0;
    *(_DWORD *)(v21 + 784) = 0;
    sub_2F8FF00(v21 + 792, v21 + 48, 0);
    v23 = (_QWORD *)(v21 + 1224);
    *(_QWORD *)(v21 + 1208) = 0;
    *(_QWORD *)(v21 + 1216) = 1;
    do
    {
      if ( v23 )
        *v23 = -4096;
      v23 += 2;
    }
    while ( v23 != (_QWORD *)(v21 + 1480) );
    if ( !byte_5038F08 && *(_BYTE *)(v21 + 632) )
    {
      v31 = *(_QWORD *)(v22 + 16);
      v32 = *(__int64 (**)())(*(_QWORD *)v31 + 128LL);
      if ( v32 == sub_2DAC790 )
        BUG();
      v33 = ((__int64 (__fastcall *)(__int64))v32)(v31);
      *(_QWORD *)(v21 + 672) = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v33 + 1032LL))(
                                 v33,
                                 v31,
                                 v21);
    }
    else
    {
      v24 = sub_22077B0(0x10u);
      if ( v24 )
      {
        *(_DWORD *)(v24 + 8) = 0;
        *(_QWORD *)v24 = &unk_4A2BCD8;
      }
      *(_QWORD *)(v21 + 672) = v24;
    }
  }
  *(_QWORD *)(v9 + 88) = v21;
  return v21;
}

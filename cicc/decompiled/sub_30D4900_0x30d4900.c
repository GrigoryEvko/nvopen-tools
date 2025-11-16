// Function: sub_30D4900
// Address: 0x30d4900
//
__int64 __fastcall sub_30D4900(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        int *a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        __int64 a13,
        char a14,
        char a15)
{
  __int64 v17; // rax
  unsigned __int8 v18; // cl
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r14
  __int64 v22; // rdx
  bool v23; // al
  _DWORD *v24; // rax
  __int64 result; // rax
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdx
  unsigned __int8 v29; // r13
  __int64 v30; // rax
  bool v31; // zf
  _DWORD *v32; // rdx
  unsigned __int8 v33; // [rsp+7h] [rbp-59h]
  unsigned __int8 v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+8h] [rbp-58h]
  _QWORD v36[2]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int8 v37; // [rsp+20h] [rbp-40h]

  *(_QWORD *)a1 = off_49D8850;
  *(_QWORD *)(a1 + 8) = a5;
  *(_QWORD *)(a1 + 16) = a7;
  *(_QWORD *)(a1 + 64) = a6;
  *(_QWORD *)(a1 + 24) = a8;
  *(_QWORD *)(a1 + 72) = a2;
  *(_QWORD *)(a1 + 32) = a9;
  *(_QWORD *)(a1 + 40) = a10;
  *(_QWORD *)(a1 + 48) = a11;
  *(_QWORD *)(a1 + 56) = a12;
  v17 = sub_B2BEC0(a2);
  *(_QWORD *)(a1 + 96) = a3;
  *(_QWORD *)(a1 + 80) = v17;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 88) = a13;
  *(_QWORD *)(a1 + 272) = a1 + 296;
  *(_BYTE *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_DWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 280) = 16;
  *(_DWORD *)(a1 + 288) = 0;
  *(_BYTE *)(a1 + 292) = 1;
  *(_QWORD *)(a1 + 424) = 0;
  *(_QWORD *)(a1 + 432) = 0;
  *(_QWORD *)(a1 + 440) = 0;
  *(_DWORD *)(a1 + 448) = 0;
  v18 = qword_502FD08;
  *(_QWORD *)(a1 + 472) = a1 + 496;
  *(_WORD *)(a1 + 456) = 1;
  *(_QWORD *)(a1 + 464) = 0;
  *(_QWORD *)(a1 + 480) = 16;
  *(_DWORD *)(a1 + 488) = 0;
  *(_BYTE *)(a1 + 492) = 1;
  *(_QWORD *)(a1 + 624) = 0;
  *(_QWORD *)(a1 + 632) = 0;
  *(_QWORD *)(a1 + 640) = 0;
  *(_QWORD *)a1 = off_49D8928;
  if ( v18 )
    goto LABEL_18;
  v19 = *(_QWORD *)(a1 + 64);
  v18 = *((_BYTE *)a4 + 61) | (a13 != 0);
  if ( v18 || !v19 || !*(_QWORD *)(v19 + 8) || !*(_QWORD *)(a1 + 32) )
    goto LABEL_3;
  v34 = *((_BYTE *)a4 + 61) | (a13 != 0);
  v26 = sub_23DF0D0(&dword_50308C8);
  v18 = v34;
  if ( v26 )
  {
    if ( !(_BYTE)qword_5030948 )
      goto LABEL_18;
  }
  else
  {
    v19 = *(_QWORD *)(a1 + 64);
    v32 = *(_DWORD **)(v19 + 8);
    if ( !v32 || *v32 )
      goto LABEL_3;
  }
  v33 = v34;
  v35 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 40LL) + 72LL);
  sub_B2EE70((__int64)v36, v35, 0);
  if ( !v37 )
  {
LABEL_34:
    v19 = *(_QWORD *)(a1 + 64);
    v18 = 0;
    goto LABEL_3;
  }
  v27 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 32))(*(_QWORD *)(a1 + 40), v35);
  v18 = v33;
  if ( !v27 )
  {
LABEL_18:
    v19 = *(_QWORD *)(a1 + 64);
    goto LABEL_3;
  }
  if ( !sub_D84510(*(_QWORD *)(a1 + 64), *(_QWORD *)(a1 + 96), v27) )
    goto LABEL_34;
  sub_B2EE70((__int64)v36, *(_QWORD *)(a1 + 72), 0);
  v29 = v37;
  if ( !v37 )
    goto LABEL_34;
  v18 = v33;
  if ( !v36[0] )
    goto LABEL_18;
  v30 = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64, _QWORD))(a1 + 32))(
          *(_QWORD *)(a1 + 40),
          *(_QWORD *)(a1 + 72),
          v28,
          v33);
  v18 = v33;
  v31 = v30 == 0;
  v19 = *(_QWORD *)(a1 + 64);
  if ( !v31 )
    v18 = v29;
LABEL_3:
  *(_QWORD *)(a1 + 672) = 0;
  v20 = *a4;
  *(_BYTE *)(a1 + 648) = v18;
  *(_DWORD *)(a1 + 652) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = a4;
  *(_QWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 696) = 0;
  *(_QWORD *)(a1 + 704) = v20;
  *(_BYTE *)(a1 + 712) = a14;
  *(_BYTE *)(a1 + 713) = a15;
  if ( !v19 || !*(_QWORD *)(v19 + 8) || !*(_QWORD *)(a1 + 32) )
    goto LABEL_16;
  if ( (unsigned int)sub_23DF0D0(&dword_50308C8) )
  {
    if ( !(_BYTE)qword_5030948 )
    {
LABEL_16:
      v23 = 0;
      goto LABEL_17;
    }
  }
  else
  {
    v24 = *(_DWORD **)(*(_QWORD *)(a1 + 64) + 8LL);
    if ( !v24 || *v24 )
      goto LABEL_16;
  }
  v21 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 96) + 40LL) + 72LL);
  sub_B2EE70((__int64)v36, v21, 0);
  if ( !v37 )
    goto LABEL_16;
  v22 = (*(__int64 (__fastcall **)(_QWORD, __int64))(a1 + 32))(*(_QWORD *)(a1 + 40), v21);
  if ( !v22 )
    goto LABEL_16;
  if ( !sub_D84510(*(_QWORD *)(a1 + 64), *(_QWORD *)(a1 + 96), v22) )
    goto LABEL_16;
  sub_B2EE70((__int64)v36, *(_QWORD *)(a1 + 72), 0);
  if ( !v37 || !v36[0] )
    goto LABEL_16;
  v23 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(a1 + 32))(*(_QWORD *)(a1 + 40), *(_QWORD *)(a1 + 72)) != 0;
LABEL_17:
  *(_BYTE *)(a1 + 714) = v23;
  *(_WORD *)(a1 + 728) = 0;
  *(_BYTE *)(a1 + 768) = 0;
  *(_QWORD *)(a1 + 792) = 0;
  *(_QWORD *)(a1 + 824) = off_4A325F8;
  *(_QWORD *)(a1 + 832) = a1;
  result = *((unsigned __int8 *)a4 + 64);
  *(_DWORD *)(a1 + 716) = 0;
  *(_QWORD *)(a1 + 720) = 0;
  *(_BYTE *)(a1 + 776) = 1;
  *(_QWORD *)(a1 + 780) = 0;
  *(_QWORD *)(a1 + 800) = 0;
  *(_QWORD *)(a1 + 808) = 0;
  *(_DWORD *)(a1 + 816) = 0;
  *(_BYTE *)(a1 + 457) = result;
  return result;
}

// Function: sub_2EC1860
// Address: 0x2ec1860
//
__int64 __fastcall sub_2EC1860(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r8
  __int64 v9; // r12
  __int64 (*v10)(); // rax
  __int64 *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rsi
  __int64 v14; // r14
  __int64 (__fastcall *v15)(__int64, unsigned __int16); // rax
  _DWORD *v16; // r15
  __int64 v17; // rdi
  void (*v18)(); // rax
  __int64 result; // rax

  v6 = sub_2E88D60(a2);
  v7 = 0;
  v8 = *(_QWORD *)(v6 + 16);
  v9 = v6;
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 144LL);
  if ( v10 != sub_2C8F680 )
    v7 = ((__int64 (__fastcall *)(__int64))v10)(v8);
  *(_BYTE *)(a1 + 32) = 1;
  v11 = (__int64 *)(v7 + 176);
  v12 = 8;
  while ( 1 )
  {
    v13 = *v11;
    if ( *v11 )
      break;
    v12 = (unsigned int)(v12 - 1);
    --v11;
    if ( (_DWORD)v12 == 2 )
      goto LABEL_10;
  }
  v14 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
  v15 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v7 + 552LL);
  if ( v15 != sub_2EC09E0 )
    v13 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v15)(v7, (unsigned int)v12, 0);
  v16 = (_DWORD *)(*(_QWORD *)v14 + 24LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL));
  if ( *(_DWORD *)(v14 + 8) != *v16 )
    sub_2F60630(v14, v13, 3LL * *(unsigned __int16 *)(*(_QWORD *)v13 + 24LL), v12);
  *(_BYTE *)(a1 + 32) = v16[1] >> 1 < a4;
LABEL_10:
  *(_BYTE *)(a1 + 35) = 1;
  v17 = *(_QWORD *)(v9 + 16);
  v18 = *(void (**)())(*(_QWORD *)v17 + 328LL);
  if ( v18 != nullsub_1609 )
    ((void (__fastcall *)(__int64, __int64, _QWORD, __int64))v18)(v17, a1 + 32, a4, v12);
  if ( !(_BYTE)qword_5021568 )
    *(_WORD *)(a1 + 32) = 0;
  result = dword_5021B48[0];
  switch ( dword_5021B48[0] )
  {
    case 1u:
      *(_WORD *)(a1 + 34) = 1;
      break;
    case 2u:
      *(_WORD *)(a1 + 34) = 256;
      break;
    case 3u:
      *(_WORD *)(a1 + 34) = 0;
      return 0;
  }
  return result;
}

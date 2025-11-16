// Function: sub_CD8610
// Address: 0xcd8610
//
__int64 __fastcall sub_CD8610(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rcx
  char v4; // al
  __int64 v5; // rcx
  char v6; // al
  __int64 v7; // rcx
  char v8; // al
  __int64 v9; // rcx
  char v10; // al
  __int64 v11; // rcx
  char v12; // al
  __int64 v13; // rcx
  char v14; // al
  _BOOL8 v15; // rcx
  char v16; // al
  __int64 v17; // rcx
  __int64 result; // rax
  char v19; // [rsp+Eh] [rbp-42h] BYREF
  char v20; // [rsp+Fh] [rbp-41h] BYREF
  __int64 v21; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v22[14]; // [rsp+18h] [rbp-38h] BYREF

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_BYTE *)a2 ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "HW1514369War",
         0,
         v3,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)a2 = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_BYTE *)(a2 + 1) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "HW3354533War",
         0,
         v5,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 1));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 1) = 0;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_BYTE *)(a2 + 2) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SW1269959War",
         0,
         v7,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 2));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 2) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_BYTE *)(a2 + 5) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ApplyLocalMemVecAccessWar",
         0,
         v9,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 5));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 5) = 0;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = *(_BYTE *)(a2 + 3) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SW977008War",
         0,
         v11,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 3));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 3) = 0;
  }
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(_BYTE *)(a2 + 4) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ApplyDivergentITexWar",
         0,
         v13,
         &v21,
         v22) )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 4));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 4) = 0;
  }
  v14 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v15 = 0;
  if ( v14 )
    v15 = *(_DWORD *)(a2 + 8) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "numTessPrfExclusionCyclesWar",
         0,
         v15,
         &v21,
         v22) )
  {
    sub_CCD060(a1, (int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_DWORD *)(a2 + 8) = 0;
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SW866285WarInfo",
         0,
         0,
         &v19,
         &v21) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "ApplyWAR",
           1,
           0,
           &v20,
           v22) )
    {
      sub_CCCDD0(a1, (_BYTE *)(a2 + 16));
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
           a1,
           "WriteWarUcode",
           1,
           0,
           &v20,
           v22) )
    {
      sub_CCC2C0(a1, (unsigned int *)(a2 + 20));
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
      v22[0] = 0;
    else
      v22[0] = *(_QWORD *)(a2 + 32);
    sub_CCC5F0(a1, (__int64)"NumBanks", v22, 1u);
    sub_CD83A0(a1, (__int64)"CbankTexBindings", (const __m128i **)(a2 + 24));
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v17 = 0;
  if ( v16 )
    v17 = *(_BYTE *)(a2 + 40) ^ 1u;
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, unsigned int *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "SW2393858War",
             0,
             v17,
             &v21,
             v22);
  if ( (_BYTE)result )
  {
    sub_CCCDD0(a1, (_BYTE *)(a2 + 40));
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, *(_QWORD *)v22);
  }
  else if ( (_BYTE)v21 )
  {
    *(_BYTE *)(a2 + 40) = 0;
  }
  return result;
}

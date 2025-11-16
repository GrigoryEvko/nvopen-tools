// Function: sub_88EAB0
// Address: 0x88eab0
//
__int64 __fastcall sub_88EAB0(__int64 *a1)
{
  __int64 v1; // r15
  __int64 v2; // rbx
  __int64 v3; // r12
  __int64 v4; // r13
  unsigned int v5; // eax
  unsigned int *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  int v10; // ecx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  int v21; // edi
  __int64 result; // rax
  int v23; // [rsp+Ch] [rbp-244h]
  int v24; // [rsp+10h] [rbp-240h]
  unsigned int v25; // [rsp+14h] [rbp-23Ch]
  __int16 v26; // [rsp+1Ah] [rbp-236h]
  unsigned int v27; // [rsp+1Ch] [rbp-234h]
  int v28; // [rsp+20h] [rbp-230h]
  unsigned __int16 v29; // [rsp+24h] [rbp-22Ch]
  __int16 v30; // [rsp+26h] [rbp-22Ah]
  int v31; // [rsp+28h] [rbp-228h]
  __int64 v32; // [rsp+28h] [rbp-228h]
  int v33; // [rsp+3Ch] [rbp-214h] BYREF
  _QWORD v34[66]; // [rsp+40h] [rbp-210h] BYREF

  v1 = *a1;
  v2 = *(_QWORD *)(*a1 + 104);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3 )
    goto LABEL_33;
  if ( (*(_BYTE *)(v2 + 28) & 1) != 0 )
  {
    sub_6854C0(0x975u, (FILE *)dword_4F07508, *a1);
LABEL_33:
    result = (__int64)sub_72C9D0();
    a1[19] = result;
    return result;
  }
  if ( *(_DWORD *)(v3 + 24) == unk_4D042F0 )
  {
    sub_6854E0(0x1C8u, *a1);
    goto LABEL_33;
  }
  v4 = *(_QWORD *)(v1 + 64);
  v27 = dword_4F063F8;
  v29 = word_4F063FC[0];
  v24 = dword_4F07508[0];
  v26 = dword_4F07508[1];
  v28 = dword_4F061D8;
  v30 = unk_4F061DC;
  v5 = sub_8D0B70(*a1);
  *(_BYTE *)(v2 + 28) |= 1u;
  ++*(_DWORD *)(v3 + 24);
  v25 = v5;
  sub_7B8190();
  v6 = (unsigned int *)qword_4F04C68;
  v7 = 776LL * dword_4F04C64;
  if ( *(_BYTE *)(v7 + qword_4F04C68[0] + 4) != 6
    || (v8 = *(_QWORD *)(v7 + qword_4F04C68[0] + 208), v31 = 0, v8 != v4)
    && (!v8
     || !v4
     || (v6 = &dword_4F07588, !dword_4F07588)
     || (v9 = *(_QWORD *)(v8 + 32), *(_QWORD *)(v4 + 32) != v9)
     || !v9) )
  {
    v23 = dword_4F04C64;
    v32 = 776LL * dword_4F04C64;
    sub_866000(v4, 1, 1u);
    v10 = v23;
    v6 = (unsigned int *)qword_4F04C68[0];
    v11 = *(int *)(qword_4F04C68[0] + v32 + 400);
    if ( (_DWORD)v11 != -1 )
    {
      v12 = *(_QWORD *)(qword_4F04C68[0] + 776 * v11 + 216);
      if ( (*(_BYTE *)(v12 + 89) & 4) != 0 )
      {
        v13 = *(_QWORD *)(*(_QWORD *)(v12 + 40) + 32LL);
        if ( v13 == v4
          || v13 && v4 && dword_4F07588 && (v14 = *(_QWORD *)(v13 + 32), *(_QWORD *)(v4 + 32) == v14) && v14 )
        {
          v15 = qword_4F04C68[0] + v32 + 776;
          v16 = dword_4F04C64;
          do
          {
            ++v10;
            *(_BYTE *)(v15 + 8) |= 0x20u;
            v15 += 776;
          }
          while ( v16 > v10 );
        }
      }
    }
    v31 = 1;
  }
  sub_7296C0(&v33);
  sub_7BC160(*(_QWORD *)(v2 + 8));
  memset(v34, 0, 0x1D8u);
  v34[19] = v34;
  v34[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v34[22]) |= 1u;
  v34[0] = v1;
  sub_63B0A0(v34);
  while ( word_4F06418[0] != 9 )
    sub_7B8B50((unsigned __int64)v34, v6, v17, v18, v19, v20);
  sub_7B8B50((unsigned __int64)v34, v6, v17, v18, v19, v20);
  *(_QWORD *)(v2 + 8) = 0;
  v21 = v33;
  dword_4F07508[0] = v24;
  LOWORD(dword_4F07508[1]) = v26;
  dword_4F063F8 = v27;
  word_4F063FC[0] = v29;
  dword_4F061D8 = v28;
  unk_4F061DC = v30;
  sub_729730(v21);
  if ( v31 )
    sub_866010();
  sub_7B8260();
  result = v25;
  if ( v25 )
    result = sub_8D0B10();
  *(_BYTE *)(v2 + 28) &= ~1u;
  --*(_DWORD *)(v3 + 24);
  return result;
}

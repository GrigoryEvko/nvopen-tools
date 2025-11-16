// Function: sub_2CBF540
// Address: 0x2cbf540
//
__int64 __fastcall sub_2CBF540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v10; // r8
  __int64 v11; // rax
  __int64 v12; // r12
  int v13; // eax
  int v14; // eax
  unsigned int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  int v19; // eax
  int v20; // eax
  unsigned int v21; // edx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v26; // [rsp+8h] [rbp-68h]
  const char *v27; // [rsp+10h] [rbp-60h] BYREF
  char v28; // [rsp+30h] [rbp-40h]
  char v29; // [rsp+31h] [rbp-3Fh]

  v10 = *(_QWORD *)(a1 + 56);
  v29 = 1;
  v26 = v10;
  v27 = "splitPhi";
  v28 = 3;
  v11 = sub_BD2DA0(80);
  v12 = v11;
  if ( v11 )
  {
    sub_B44260(v11, a2, 55, 0x8000000u, v26, 1u);
    *(_DWORD *)(v12 + 72) = 2;
    sub_BD6B50((unsigned __int8 *)v12, &v27);
    sub_BD2A10(v12, *(_DWORD *)(v12 + 72), 1);
  }
  v13 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  if ( v13 == *(_DWORD *)(v12 + 72) )
  {
    sub_B48D90(v12);
    v13 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  }
  v14 = (v13 + 1) & 0x7FFFFFF;
  v15 = v14 | *(_DWORD *)(v12 + 4) & 0xF8000000;
  v16 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v14 - 1);
  *(_DWORD *)(v12 + 4) = v15;
  if ( *(_QWORD *)v16 )
  {
    v17 = *(_QWORD *)(v16 + 8);
    **(_QWORD **)(v16 + 16) = v17;
    if ( v17 )
      *(_QWORD *)(v17 + 16) = *(_QWORD *)(v16 + 16);
  }
  *(_QWORD *)v16 = a3;
  if ( a3 )
  {
    v18 = *(_QWORD *)(a3 + 16);
    *(_QWORD *)(v16 + 8) = v18;
    if ( v18 )
      *(_QWORD *)(v18 + 16) = v16 + 8;
    *(_QWORD *)(v16 + 16) = a3 + 16;
    *(_QWORD *)(a3 + 16) = v16;
  }
  *(_QWORD *)(*(_QWORD *)(v12 - 8) + 32LL * *(unsigned int *)(v12 + 72)
                                   + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = a4;
  v19 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  if ( v19 == *(_DWORD *)(v12 + 72) )
  {
    sub_B48D90(v12);
    v19 = *(_DWORD *)(v12 + 4) & 0x7FFFFFF;
  }
  v20 = (v19 + 1) & 0x7FFFFFF;
  v21 = v20 | *(_DWORD *)(v12 + 4) & 0xF8000000;
  v22 = *(_QWORD *)(v12 - 8) + 32LL * (unsigned int)(v20 - 1);
  *(_DWORD *)(v12 + 4) = v21;
  if ( *(_QWORD *)v22 )
  {
    v23 = *(_QWORD *)(v22 + 8);
    **(_QWORD **)(v22 + 16) = v23;
    if ( v23 )
      *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
  }
  *(_QWORD *)v22 = a5;
  if ( a5 )
  {
    v24 = *(_QWORD *)(a5 + 16);
    *(_QWORD *)(v22 + 8) = v24;
    if ( v24 )
      *(_QWORD *)(v24 + 16) = v22 + 8;
    *(_QWORD *)(v22 + 16) = a5 + 16;
    *(_QWORD *)(a5 + 16) = v22;
  }
  *(_QWORD *)(*(_QWORD *)(v12 - 8) + 32LL * *(unsigned int *)(v12 + 72)
                                   + 8LL * ((*(_DWORD *)(v12 + 4) & 0x7FFFFFFu) - 1)) = a6;
  return v12;
}

// Function: sub_37C7180
// Address: 0x37c7180
//
__int64 __fastcall sub_37C7180(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 *v6; // rax
  unsigned __int64 v7; // rsi
  __int64 v8; // r12
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  __int64 v11; // rsi
  unsigned __int16 v12; // di
  int v13; // r10d
  unsigned int i; // eax
  __int64 v15; // rcx
  unsigned int v16; // eax
  __int64 v17; // rdx
  int v18; // eax
  char v19; // dl
  unsigned __int64 v20; // rax
  char v21; // r8
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // rsi
  int v25; // edx
  bool v26; // zf
  __int64 v27; // [rsp+8h] [rbp-28h]
  __int64 v28; // [rsp+10h] [rbp-20h] BYREF
  char v29; // [rsp+18h] [rbp-18h]

  v27 = sub_37C70E0(a1, a2);
  if ( !BYTE4(v27) )
    goto LABEL_2;
  v6 = (__int64 *)sub_2E864A0(a2);
  v4 = 0x3FFFFFFFFFFFFFFFLL;
  v5 = *v6;
  LOBYTE(v6) = 1;
  v7 = *(_QWORD *)(v5 + 24);
  if ( (v7 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
  {
    v19 = *(_BYTE *)(v5 + 24);
    v20 = v7 >> 3;
    v21 = v19 & 2;
    if ( (v19 & 6) == 2 || (*(_BYTE *)(v5 + 24) & 1) != 0 )
    {
      v22 = HIDWORD(v7);
      if ( v21 )
        v22 = HIWORD(v7);
      v4 = v22;
    }
    else
    {
      v23 = *(_QWORD *)(v5 + 24);
      v24 = HIDWORD(v7);
      v25 = HIWORD(v23);
      if ( !v21 )
        v25 = v24;
      v4 = (unsigned int)(unsigned __int16)((unsigned int)v23 >> 8) * v25;
      v26 = (v20 & 1) == 0;
      v22 = v4 | 0x4000000000000000LL;
      if ( v26 )
        v22 = v4;
    }
    v6 = (__int64 *)(v22 >> 62);
  }
  v8 = *(_QWORD *)(a1 + 408);
  v28 = v4;
  v29 = (char)v6;
  v9 = sub_CA1930(&v28);
  v10 = *(unsigned int *)(v8 + 848);
  v11 = *(_QWORD *)(v8 + 832);
  v12 = v9;
  if ( (_DWORD)v10 )
  {
    v13 = 1;
    for ( i = (v10 - 1) & (1512728442 * v9); ; i = (v10 - 1) & v16 )
    {
      v15 = v11 + 8LL * i;
      if ( v12 == *(_WORD *)v15 && !*(_WORD *)(v15 + 2) )
        break;
      if ( *(_WORD *)v15 == 0xFFFF && *(_WORD *)(v15 + 2) == 0xFFFF )
        goto LABEL_11;
      v16 = v13 + i;
      ++v13;
    }
  }
  else
  {
LABEL_11:
    v15 = v11 + 8 * v10;
  }
  v17 = *(_QWORD *)(a1 + 408);
  if ( v15 == *(_QWORD *)(v17 + 832) + 8LL * *(unsigned int *)(v17 + 848) )
  {
LABEL_2:
    BYTE4(v28) = 0;
    return v28;
  }
  else
  {
    v18 = *(_DWORD *)(*(_QWORD *)(v17 + 64)
                    + 4LL
                    * (unsigned int)(*(_DWORD *)(v15 + 4) + *(_DWORD *)(v17 + 284) + *(_DWORD *)(v17 + 288) * (v27 - 1)));
    BYTE4(v28) = 1;
    LODWORD(v28) = v18;
    return v28;
  }
}

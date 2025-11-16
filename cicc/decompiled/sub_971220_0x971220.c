// Function: sub_971220
// Address: 0x971220
//
__int64 __fastcall sub_971220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  unsigned __int64 v7; // r15
  char v8; // dl
  unsigned __int64 v9; // rax
  char v10; // dl
  __int64 v11; // r8
  int v13; // edx
  __int64 v14; // r15
  unsigned __int64 v15; // rbx
  char v16; // dl
  __int64 v17; // rax
  char v18; // dl
  __int64 v19; // rcx
  int v20; // edx
  __int64 *v21; // rax
  char v22; // r15
  __int64 v23; // rax
  __int64 v24; // rcx
  int v25; // edx
  __int64 *v26; // rax
  char v27; // al
  __int64 v28; // rax
  char v29; // al
  unsigned int v30; // r15d
  char v31; // al
  __int64 i; // rsi
  __int64 v33; // rax
  __int64 v34; // rbx
  char v35; // [rsp+Fh] [rbp-71h]
  char v36; // [rsp+10h] [rbp-70h]
  char v37; // [rsp+18h] [rbp-68h]
  __int64 v38; // [rsp+18h] [rbp-68h]
  __int64 v39; // [rsp+18h] [rbp-68h]
  __int64 v40; // [rsp+18h] [rbp-68h]
  unsigned __int64 v41; // [rsp+30h] [rbp-50h]

  while ( 1 )
  {
    v6 = *(_QWORD *)(a1 + 8);
    if ( a2 == v6 )
      return a1;
    v7 = sub_9208B0(a3, a2);
    v37 = v8;
    v9 = sub_9208B0(a3, v6);
    v41 = v9;
    if ( !v10 && v37 )
      return 0;
    v35 = v10;
    if ( v9 < v7 )
      return 0;
    v11 = sub_96E500((unsigned __int8 *)a1, a2, a3);
    if ( v11 )
      return v11;
    if ( v41 == v7 && v37 == v35 )
    {
      v19 = v6;
      v20 = *(unsigned __int8 *)(v6 + 8);
      if ( (unsigned int)(v20 - 17) <= 1 )
      {
        v21 = *(__int64 **)(v6 + 16);
        v19 = *v21;
        LOBYTE(v20) = *(_BYTE *)(*v21 + 8);
      }
      v22 = 0;
      if ( (_BYTE)v20 == 14 )
      {
        v23 = sub_AE2980(a3, *(_DWORD *)(v19 + 8) >> 8);
        v11 = 0;
        v22 = *(_BYTE *)(v23 + 16);
      }
      v24 = a2;
      v25 = *(unsigned __int8 *)(a2 + 8);
      if ( (unsigned int)(v25 - 17) <= 1 )
      {
        v26 = *(__int64 **)(a2 + 16);
        v24 = *v26;
        LOBYTE(v25) = *(_BYTE *)(*v26 + 8);
      }
      v27 = 0;
      if ( (_BYTE)v25 == 14 )
      {
        v39 = v11;
        v28 = sub_AE2980(a3, *(_DWORD *)(v24 + 8) >> 8);
        v11 = v39;
        v27 = *(_BYTE *)(v28 + 16);
      }
      if ( v27 == v22 )
      {
        v29 = *(_BYTE *)(v6 + 8);
        if ( v29 == 12 )
        {
          v30 = (*(_BYTE *)(a2 + 8) != 14) + 48;
        }
        else
        {
          v30 = 49;
          if ( v29 == 14 )
            v30 = 2 * (*(_BYTE *)(a2 + 8) != 12) + 47;
        }
        v40 = v11;
        v31 = sub_B50F30(v30, *(_QWORD *)(a1 + 8), a2, v24);
        v11 = v40;
        if ( v31 )
          break;
      }
    }
    v13 = *(unsigned __int8 *)(v6 + 8);
    if ( (_BYTE)v13 == 15 )
    {
      for ( i = 0; ; i = (unsigned int)(i + 1) )
      {
        v33 = sub_AD69F0(a1, i);
        v34 = v33;
        if ( !v33 || sub_9208B0(a3, *(_QWORD *)(v33 + 8)) )
          break;
      }
      a1 = v34;
    }
    else
    {
      if ( (unsigned __int8)(*(_BYTE *)(v6 + 8) - 16) > 2u )
        return 0;
      v38 = v11;
      if ( (unsigned int)(v13 - 17) <= 1 )
      {
        v14 = *(_QWORD *)(v6 + 24);
        v15 = (sub_9208B0(a3, v14) + 7) & 0xFFFFFFFFFFFFFFF8LL;
        v36 = v16;
        v17 = sub_9208B0(a3, v14);
        v11 = v38;
        if ( v17 != v15 || v18 != v36 )
          return v11;
      }
      a1 = sub_AD69F0(a1, 0);
    }
    if ( !a1 )
      return 0;
  }
  return sub_96F480(v30, a1, a2, a3);
}

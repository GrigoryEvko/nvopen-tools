// Function: sub_2896320
// Address: 0x2896320
//
void __fastcall sub_2896320(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // rax
  __int64 v6; // rsi
  unsigned int v8; // edx
  __int64 *v9; // rcx
  __int64 v10; // rdi
  int v11; // edx
  int v12; // eax
  int v13; // esi
  unsigned __int8 *v14; // rsi
  unsigned int v15; // esi
  __int64 v16; // r10
  unsigned int v17; // ecx
  unsigned __int8 **v18; // r9
  unsigned __int8 *v19; // r8
  __int64 v20; // rsi
  int v21; // ecx
  int v22; // r9d
  int v23; // r15d
  unsigned __int8 **v24; // rdi
  unsigned int v25; // ecx
  __int64 v26; // rax
  unsigned __int8 **v27; // [rsp+18h] [rbp-58h] BYREF
  unsigned __int8 *v28; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int8 *v29; // [rsp+28h] [rbp-48h]
  int v30; // [rsp+30h] [rbp-40h]

  v5 = *(unsigned int *)(a1 + 88);
  v6 = *(_QWORD *)(a1 + 72);
  if ( !(_DWORD)v5 )
    goto LABEL_10;
  v8 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (__int64 *)(v6 + 24LL * v8);
  v10 = *v9;
  if ( a2 == *v9 )
  {
LABEL_3:
    if ( v9 == (__int64 *)(v6 + 24 * v5) )
      goto LABEL_10;
    *v9 = -8192;
    v11 = *(_DWORD *)(a1 + 80);
    *(_DWORD *)(a1 + 80) = v11 - 1;
    v12 = *(_DWORD *)(a1 + 84) + 1;
    *(_DWORD *)(a1 + 84) = v12;
    v13 = *a3;
    if ( (unsigned __int8)v13 <= 0x1Cu )
      goto LABEL_10;
    if ( (_BYTE)v13 == 85 )
    {
      v20 = *((_QWORD *)a3 - 4);
      if ( !v20
        || *(_BYTE *)v20
        || *(_QWORD *)(v20 + 24) != *((_QWORD *)a3 + 10)
        || (*(_BYTE *)(v20 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v20 + 36) - 231) > 3 )
      {
        goto LABEL_10;
      }
    }
    else if ( (unsigned __int8)(v13 - 61) > 1u && (unsigned int)(v13 - 41) > 6 )
    {
      goto LABEL_10;
    }
    v14 = (unsigned __int8 *)v9[1];
    v28 = a3;
    v29 = v14;
    v15 = *(_DWORD *)(a1 + 88);
    v30 = *((_DWORD *)v9 + 4);
    if ( v15 )
    {
      v16 = *(_QWORD *)(a1 + 72);
      v17 = (v15 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v18 = (unsigned __int8 **)(v16 + 24LL * v17);
      v19 = *v18;
      if ( a3 == *v18 )
        goto LABEL_10;
      v23 = 1;
      v24 = 0;
      while ( v19 != (unsigned __int8 *)-4096LL )
      {
        if ( v19 != (unsigned __int8 *)-8192LL || v24 )
          v18 = v24;
        v17 = (v15 - 1) & (v23 + v17);
        v19 = *(unsigned __int8 **)(v16 + 24LL * v17);
        if ( a3 == v19 )
          goto LABEL_10;
        ++v23;
        v24 = v18;
        v18 = (unsigned __int8 **)(v16 + 24LL * v17);
      }
      if ( !v24 )
        v24 = v18;
      ++*(_QWORD *)(a1 + 64);
      v27 = v24;
      if ( 4 * v11 < 3 * v15 )
      {
        v25 = v15 - v11 - v12;
        v26 = (__int64)a3;
        if ( v25 > v15 >> 3 )
        {
LABEL_27:
          *(_DWORD *)(a1 + 80) = v11;
          if ( *v24 != (unsigned __int8 *)-4096LL )
            --*(_DWORD *)(a1 + 84);
          *v24 = (unsigned __int8 *)v26;
          v24[1] = v29;
          *((_DWORD *)v24 + 4) = v30;
          goto LABEL_10;
        }
LABEL_32:
        sub_2895430(a1 + 64, v15);
        sub_28941A0(a1 + 64, (__int64 *)&v28, &v27);
        v26 = (__int64)v28;
        v11 = *(_DWORD *)(a1 + 80) + 1;
        v24 = v27;
        goto LABEL_27;
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 64);
      v27 = 0;
    }
    v15 *= 2;
    goto LABEL_32;
  }
  v21 = 1;
  while ( v10 != -4096 )
  {
    v22 = v21 + 1;
    v8 = (v5 - 1) & (v21 + v8);
    v9 = (__int64 *)(v6 + 24LL * v8);
    v10 = *v9;
    if ( a2 == *v9 )
      goto LABEL_3;
    v21 = v22;
  }
LABEL_10:
  sub_BD84D0(a2, (__int64)a3);
}

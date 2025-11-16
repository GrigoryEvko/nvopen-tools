// Function: sub_2571D60
// Address: 0x2571d60
//
__int64 __fastcall sub_2571D60(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  __int64 v6; // rcx
  __int64 *v7; // rbx
  __int64 *v8; // r13
  _QWORD *v9; // rdi
  _QWORD *v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rax
  _BYTE *v16; // r8
  int v17; // eax
  unsigned int v18; // esi
  int v19; // r15d
  __int64 v20; // r9
  _QWORD *v21; // r11
  unsigned int v22; // edx
  _QWORD *v23; // rdi
  _BYTE *v24; // rcx
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // r15
  __int64 result; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 *i; // rbx
  __int64 *v33; // rsi
  _BYTE *v34; // [rsp+10h] [rbp-40h] BYREF
  _QWORD *v35; // [rsp+18h] [rbp-38h] BYREF

  v3 = sub_2509740((_QWORD *)(a1 + 72));
  if ( (*(_BYTE *)(v3 + 7) & 0x20) != 0 )
  {
    v4 = sub_B91C10(v3, 23);
    if ( v4 )
    {
      v5 = *(_BYTE *)(v4 - 16);
      if ( (v5 & 2) != 0 )
      {
        v7 = *(__int64 **)(v4 - 32);
        v6 = *(unsigned int *)(v4 - 24);
      }
      else
      {
        v6 = (*(_WORD *)(v4 - 16) >> 6) & 0xF;
        v7 = (__int64 *)(v4 - 8LL * ((v5 >> 2) & 0xF) - 16);
      }
      v8 = &v7[v6];
      if ( v8 == v7 )
        goto LABEL_26;
      while ( 1 )
      {
        v15 = *v7;
        if ( !*v7 )
          goto LABEL_9;
        if ( *(_BYTE *)v15 != 1 )
          goto LABEL_9;
        v16 = *(_BYTE **)(v15 + 136);
        if ( *v16 )
          goto LABEL_9;
        v17 = *(_DWORD *)(a1 + 152);
        v34 = v16;
        if ( !v17 )
        {
          v9 = *(_QWORD **)(a1 + 168);
          v10 = &v9[*(unsigned int *)(a1 + 176)];
          if ( v10 == sub_2538080(v9, (__int64)v10, (__int64 *)&v34) )
            sub_25718D0(a1 + 136, v13, v11, v12, v13, v14);
          goto LABEL_9;
        }
        v18 = *(_DWORD *)(a1 + 160);
        if ( !v18 )
        {
          ++*(_QWORD *)(a1 + 136);
          v35 = 0;
LABEL_36:
          v18 *= 2;
LABEL_37:
          sub_A35F10(a1 + 136, v18);
          sub_A2AFD0(a1 + 136, (__int64 *)&v34, &v35);
          v16 = v34;
          v21 = v35;
          v25 = *(_DWORD *)(a1 + 152) + 1;
          goto LABEL_21;
        }
        v19 = 1;
        v20 = *(_QWORD *)(a1 + 144);
        v21 = 0;
        v22 = (v18 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v23 = (_QWORD *)(v20 + 8LL * v22);
        v24 = (_BYTE *)*v23;
        if ( v16 == (_BYTE *)*v23 )
        {
LABEL_9:
          if ( v8 == ++v7 )
            goto LABEL_26;
        }
        else
        {
          while ( v24 != (_BYTE *)-4096LL )
          {
            if ( v24 != (_BYTE *)-8192LL || v21 )
              v23 = v21;
            v22 = (v18 - 1) & (v19 + v22);
            v24 = *(_BYTE **)(v20 + 8LL * v22);
            if ( v16 == v24 )
              goto LABEL_9;
            ++v19;
            v21 = v23;
            v23 = (_QWORD *)(v20 + 8LL * v22);
          }
          if ( !v21 )
            v21 = v23;
          v25 = v17 + 1;
          ++*(_QWORD *)(a1 + 136);
          v35 = v21;
          if ( 4 * v25 >= 3 * v18 )
            goto LABEL_36;
          if ( v18 - *(_DWORD *)(a1 + 156) - v25 <= v18 >> 3 )
            goto LABEL_37;
LABEL_21:
          *(_DWORD *)(a1 + 152) = v25;
          if ( *v21 != -4096 )
            --*(_DWORD *)(a1 + 156);
          *v21 = v16;
          v26 = *(unsigned int *)(a1 + 176);
          v27 = (__int64)v34;
          if ( v26 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 180) )
          {
            sub_C8D5F0(a1 + 168, (const void *)(a1 + 184), v26 + 1, 8u, (__int64)v16, v20);
            v26 = *(unsigned int *)(a1 + 176);
          }
          ++v7;
          *(_QWORD *)(*(_QWORD *)(a1 + 168) + 8 * v26) = v27;
          ++*(_DWORD *)(a1 + 176);
          if ( v8 == v7 )
            goto LABEL_26;
        }
      }
    }
  }
  result = sub_250EEA0(a2);
  if ( !(_BYTE)result )
    return result;
  if ( (unsigned __int8)sub_250EEA0(a2) )
  {
    v29 = sub_250ED30(*(_QWORD *)(a2 + 208));
    v31 = v29 + 8 * v30;
    for ( i = (__int64 *)v29; i != (__int64 *)v31; ++i )
    {
      v33 = i;
      sub_2571A80(a1 + 136, v33);
    }
  }
LABEL_26:
  result = *(unsigned int *)(a1 + 176);
  if ( !(_DWORD)result )
  {
    result = *(unsigned __int8 *)(a1 + 97);
    *(_BYTE *)(a1 + 96) = result;
  }
  return result;
}

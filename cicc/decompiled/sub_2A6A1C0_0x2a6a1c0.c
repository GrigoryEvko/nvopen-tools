// Function: sub_2A6A1C0
// Address: 0x2a6a1c0
//
unsigned __int8 *__fastcall sub_2A6A1C0(__int64 a1, unsigned __int8 *a2, unsigned int a3)
{
  __int64 v3; // r15
  unsigned int v7; // esi
  int v8; // r10d
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int i; // eax
  __int64 v12; // r14
  unsigned __int8 *v13; // rdi
  unsigned int v14; // eax
  unsigned __int8 *v15; // r14
  int v17; // eax
  int v18; // edi
  __int64 v19; // rdx
  unsigned __int8 *v20; // rsi
  __int64 v21; // [rsp+8h] [rbp-B8h]
  __int64 v22; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int8 v23[48]; // [rsp+20h] [rbp-A0h] BYREF
  unsigned __int8 *v24; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v25; // [rsp+58h] [rbp-68h]
  unsigned __int8 v26[96]; // [rsp+60h] [rbp-60h] BYREF

  v3 = a1 + 168;
  v24 = a2;
  v7 = *(_DWORD *)(a1 + 192);
  *(_QWORD *)v23 = 0;
  v25 = a3;
  *(_QWORD *)v26 = 0;
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 168);
    v22 = 0;
    goto LABEL_28;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 176);
  v10 = 0;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * ((37 * a3) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))) >> 31)
           ^ (756364221 * a3)); ; i = (v7 - 1) & v14 )
  {
    v12 = v9 + 56LL * i;
    v13 = *(unsigned __int8 **)v12;
    if ( a2 == *(unsigned __int8 **)v12 && a3 == *(_DWORD *)(v12 + 8) )
    {
      v15 = (unsigned __int8 *)(v12 + 16);
      sub_22C0090(v26);
      sub_22C0090(v23);
      return v15;
    }
    if ( v13 == (unsigned __int8 *)-4096LL )
      break;
    if ( v13 == (unsigned __int8 *)-8192LL && *(_DWORD *)(v12 + 8) == -2 && !v10 )
      v10 = v9 + 56LL * i;
LABEL_9:
    v14 = v8 + i;
    ++v8;
  }
  if ( *(_DWORD *)(v12 + 8) != -1 )
    goto LABEL_9;
  v17 = *(_DWORD *)(a1 + 184);
  if ( !v10 )
    v10 = v12;
  ++*(_QWORD *)(a1 + 168);
  v18 = v17 + 1;
  v22 = v10;
  if ( 4 * (v17 + 1) < 3 * v7 )
  {
    v19 = (__int64)a2;
    if ( v7 - *(_DWORD *)(a1 + 188) - v18 > v7 >> 3 )
      goto LABEL_18;
    goto LABEL_29;
  }
LABEL_28:
  v7 *= 2;
LABEL_29:
  sub_2A69E40(v3, v7);
  sub_2A65F30(v3, (__int64 *)&v24, &v22);
  v19 = (__int64)v24;
  v10 = v22;
  v18 = *(_DWORD *)(a1 + 184) + 1;
LABEL_18:
  *(_DWORD *)(a1 + 184) = v18;
  if ( *(_QWORD *)v10 != -4096 || *(_DWORD *)(v10 + 8) != -1 )
    --*(_DWORD *)(a1 + 188);
  *(_QWORD *)v10 = v19;
  v15 = (unsigned __int8 *)(v10 + 16);
  v21 = v10;
  *(_DWORD *)(v10 + 8) = v25;
  sub_22C0650(v10 + 16, v26);
  sub_22C0090(v26);
  sub_22C0090(v23);
  if ( *a2 <= 0x15u )
  {
    v20 = (unsigned __int8 *)sub_AD69F0(a2, a3);
    if ( v20 )
    {
      sub_2A624B0((__int64)v15, v20, 0);
    }
    else if ( *(_BYTE *)(v21 + 16) != 6 )
    {
      sub_22C0090(v15);
      *(_BYTE *)(v21 + 16) = 6;
    }
  }
  return v15;
}

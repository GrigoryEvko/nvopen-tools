// Function: sub_1BA7710
// Address: 0x1ba7710
//
unsigned __int64 __fastcall sub_1BA7710(__int64 a1, __int64 a2, int a3)
{
  __int64 **v3; // r13
  __int64 *v5; // rdi
  _BOOL4 v6; // ebx
  unsigned int v7; // edx
  char v8; // al
  __int64 *v9; // rdi
  unsigned int v10; // ebx
  bool v11; // al
  int v13; // eax
  char v14; // al
  __int64 *v15; // r13
  unsigned int v16; // esi
  int v17; // eax
  int v18; // eax
  unsigned int v19; // eax
  __int64 *v20; // r12
  __int64 v21; // rax
  unsigned int v22; // esi
  __int64 *v23; // r9
  __int64 v24; // rcx
  int v25; // edx
  unsigned int v26; // eax
  __int64 v27; // rdi
  unsigned int v28; // edx
  __int64 v29; // rax
  __int64 v30; // r11
  int v31; // r15d
  int v32; // eax
  unsigned int v33; // [rsp+14h] [rbp-4Ch] BYREF
  __int64 **v34; // [rsp+18h] [rbp-48h] BYREF
  int v35; // [rsp+24h] [rbp-3Ch] BYREF
  __int64 *v36[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = (__int64 **)a2;
  v34 = (__int64 **)a2;
  v33 = a3;
  v35 = a3;
  if ( a3 == 1 )
    goto LABEL_18;
  if ( (unsigned __int8)sub_1B97860(a1 + 168, &v35, v36) )
    v5 = v36[0];
  else
    v5 = (__int64 *)(*(_QWORD *)(a1 + 176) + 80LL * *(unsigned int *)(a1 + 192));
  v6 = sub_13A0E30((__int64)(v5 + 1), a2);
  if ( v6 )
  {
    v3 = v34;
LABEL_18:
    v33 = 1;
    v7 = 1;
    goto LABEL_12;
  }
  v7 = v33;
  v3 = v34;
  if ( v33 <= 1 )
  {
LABEL_12:
    v10 = sub_1BA2D60(a1, v3, v7, v36);
    v11 = 0;
    if ( v33 > 1 )
    {
      v11 = 0;
      if ( *((_BYTE *)v36[0] + 8) == 16 )
      {
        v26 = sub_14A35F0(*(_QWORD *)(a1 + 328));
        v11 = v26 < v33;
      }
    }
    return ((unsigned __int64)v11 << 32) | v10;
  }
  if ( !(unsigned __int8)sub_1B918B0(a1, (__int64)v34, v33) )
  {
    if ( (unsigned __int8)sub_1B97860(a1 + 232, (int *)&v33, v36) )
    {
      v8 = sub_1B97860(a1 + 232, (int *)&v33, v36);
      v9 = v36[0];
      if ( !v8 )
        v9 = (__int64 *)(*(_QWORD *)(a1 + 240) + 80LL * *(unsigned int *)(a1 + 256));
      if ( sub_13A0E30((__int64)(v9 + 1), (__int64)v34) )
      {
        v13 = sub_1BA7710(a1, v34, 1);
        return v33 * v13;
      }
    }
    v7 = v33;
    v3 = v34;
    goto LABEL_12;
  }
  v14 = sub_1B977C0(a1 + 136, (int *)&v33, v36);
  v15 = v36[0];
  if ( !v14 )
  {
    v16 = *(_DWORD *)(a1 + 160);
    v17 = *(_DWORD *)(a1 + 152);
    ++*(_QWORD *)(a1 + 136);
    v18 = v17 + 1;
    if ( 4 * v18 >= 3 * v16 )
    {
      v16 *= 2;
    }
    else if ( v16 - *(_DWORD *)(a1 + 156) - v18 > v16 >> 3 )
    {
LABEL_23:
      *(_DWORD *)(a1 + 152) = v18;
      if ( *(_DWORD *)v15 != -1 )
        --*(_DWORD *)(a1 + 156);
      v19 = v33;
      v15[1] = 0;
      v20 = v15 + 1;
      v15[2] = 0;
      v15[3] = 0;
      *((_DWORD *)v15 + 8) = 0;
      *(_DWORD *)v15 = v19;
      v21 = 1;
      goto LABEL_26;
    }
    sub_1BA7500(a1 + 136, v16);
    sub_1B977C0(a1 + 136, (int *)&v33, v36);
    v15 = v36[0];
    v18 = *(_DWORD *)(a1 + 152) + 1;
    goto LABEL_23;
  }
  v22 = *((_DWORD *)v36[0] + 8);
  v27 = v36[0][2];
  v20 = v36[0] + 1;
  if ( !v22 )
  {
    v21 = v36[0][1] + 1;
LABEL_26:
    v15[1] = v21;
    v22 = 0;
    goto LABEL_27;
  }
  v24 = (__int64)v34;
  v28 = (v22 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
  v29 = v27 + 16LL * v28;
  v30 = *(_QWORD *)v29;
  if ( v34 == *(__int64 ***)v29 )
    return *(unsigned int *)(v29 + 8);
  v31 = 1;
  v23 = 0;
  while ( v30 != -8 )
  {
    if ( v30 == -16 && !v23 )
      v23 = (__int64 *)v29;
    v28 = (v22 - 1) & (v31 + v28);
    v29 = v27 + 16LL * v28;
    v30 = *(_QWORD *)v29;
    if ( v34 == *(__int64 ***)v29 )
      return *(unsigned int *)(v29 + 8);
    ++v31;
  }
  if ( !v23 )
    v23 = (__int64 *)v29;
  v32 = *((_DWORD *)v36[0] + 6);
  ++v36[0][1];
  v25 = v32 + 1;
  if ( 4 * (v32 + 1) >= 3 * v22 )
  {
    v22 *= 2;
  }
  else if ( v22 - *((_DWORD *)v15 + 7) - v25 > v22 >> 3 )
  {
    goto LABEL_28;
  }
LABEL_27:
  sub_14672C0((__int64)v20, v22);
  sub_1463AD0((__int64)v20, (__int64 *)&v34, v36);
  v23 = v36[0];
  v24 = (__int64)v34;
  v25 = *((_DWORD *)v15 + 6) + 1;
LABEL_28:
  *((_DWORD *)v15 + 6) = v25;
  if ( *v23 != -8 )
    --*((_DWORD *)v15 + 7);
  *v23 = v24;
  *((_DWORD *)v23 + 2) = 0;
  return v6;
}

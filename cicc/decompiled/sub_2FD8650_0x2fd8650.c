// Function: sub_2FD8650
// Address: 0x2fd8650
//
void __fastcall sub_2FD8650(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int *a6,
        __int64 a7,
        char a8)
{
  _DWORD *v12; // rdx
  int v13; // ecx
  int v14; // r15d
  unsigned int v15; // r12d
  unsigned int v16; // r11d
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // eax
  unsigned int *v20; // r9
  __int64 v21; // r8
  __int64 v22; // rax
  unsigned __int64 v23; // rcx
  __int64 v24; // rax
  __int32 v25; // r8d
  __int64 v26; // rcx
  int v27; // eax
  int v28; // edx
  int v29; // edi
  unsigned int v30; // eax
  int v31; // esi
  unsigned __int64 v33; // [rsp+10h] [rbp-90h]
  int v34; // [rsp+18h] [rbp-88h]
  unsigned int v35; // [rsp+20h] [rbp-80h]
  int v37; // [rsp+34h] [rbp-6Ch] BYREF
  unsigned __int64 v38; // [rsp+38h] [rbp-68h] BYREF
  unsigned __int64 v39; // [rsp+40h] [rbp-60h] BYREF
  int v40; // [rsp+48h] [rbp-58h]

  v12 = *(_DWORD **)(a2 + 32);
  v13 = *(_DWORD *)(a2 + 40) & 0xFFFFFF;
  v14 = v12[2];
  if ( v13 == 1 )
  {
LABEL_6:
    v16 = v12[2];
    v15 = 0;
  }
  else
  {
    v15 = 1;
    while ( a4 != *(_QWORD *)&v12[10 * v15 + 16] )
    {
      v15 += 2;
      if ( v13 == v15 )
        goto LABEL_6;
    }
    v12 += 10 * v15;
    v16 = v12[2];
  }
  v35 = v16;
  v34 = (*v12 >> 8) & 0xFFF;
  v17 = *(_QWORD *)(*(_QWORD *)(a1[3] + 56) + 16LL * (v14 & 0x7FFFFFFF));
  v38 = __PAIR64__(v34, v16);
  v37 = v14;
  v33 = v17 & 0xFFFFFFFFFFFFFFF8LL;
  sub_2FD83C0((__int64)&v39, a5, &v37, (__int64 *)&v38);
  v19 = sub_2EC06C0(a1[3], v33, byte_3F871B3, 0, v33, v18);
  v20 = a6;
  v39 = __PAIR64__(v35, v19);
  v21 = v19;
  v22 = a6[2];
  v23 = a6[3];
  v40 = v34;
  if ( v22 + 1 > v23 )
  {
    sub_C8D5F0((__int64)a6, a6 + 4, v22 + 1, 0xCu, v21, (__int64)a6);
    v20 = a6;
    v22 = a6[2];
  }
  v24 = *(_QWORD *)v20 + 12 * v22;
  *(_QWORD *)v24 = v39;
  *(_DWORD *)(v24 + 8) = v40;
  ++v20[2];
  if ( (unsigned __int8)sub_2FD5CE0(v14, a3, a1[3]) )
    goto LABEL_12;
  v26 = *(_QWORD *)(a7 + 8);
  v27 = *(_DWORD *)(a7 + 24);
  if ( v27 )
  {
    v28 = v27 - 1;
    v29 = 1;
    v30 = (v27 - 1) & (37 * v14);
    v31 = *(_DWORD *)(v26 + 4LL * v30);
    if ( v14 == v31 )
    {
LABEL_12:
      sub_2FD7D90((__int64)a1, v14, v25, a4);
      goto LABEL_13;
    }
    while ( v31 != -1 )
    {
      v30 = v28 & (v29 + v30);
      v31 = *(_DWORD *)(v26 + 4LL * v30);
      if ( v14 == v31 )
        goto LABEL_12;
      ++v29;
    }
  }
LABEL_13:
  if ( a8 )
  {
    sub_2E8A650(a2, v15 + 1);
    sub_2E8A650(a2, v15);
    if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) == 1 )
    {
      if ( *(_BYTE *)(a3 + 217) || *(_QWORD *)(a3 + 224) )
        sub_2E88D70(a2, (unsigned __int16 *)(*(_QWORD *)(*a1 + 8) - 400LL));
      else
        sub_2E88E20(a2);
    }
  }
}

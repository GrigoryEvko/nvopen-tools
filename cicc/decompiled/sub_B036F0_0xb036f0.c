// Function: sub_B036F0
// Address: 0xb036f0
//
__int64 __fastcall sub_B036F0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int a6, char a7)
{
  __int64 v7; // r10
  unsigned int v10; // r12d
  __int64 v12; // r9
  int v13; // r14d
  __int64 v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // rsi
  __int64 v17; // rax
  int v18; // eax
  __int64 v19; // r9
  __int64 v20; // r8
  int v21; // r11d
  int v22; // ecx
  unsigned int v23; // r9d
  unsigned int i; // esi
  _QWORD *v25; // rdi
  unsigned int v26; // esi
  __int64 v27; // r14
  unsigned __int8 v28; // al
  __int64 result; // rax
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // r14
  _BYTE *v34; // rax
  _BYTE *v35; // rax
  _BYTE *v36; // rax
  __int64 v37; // [rsp+0h] [rbp-A0h]
  int v38; // [rsp+8h] [rbp-98h]
  unsigned int v39; // [rsp+10h] [rbp-90h]
  __int64 v40; // [rsp+18h] [rbp-88h]
  __int64 v41; // [rsp+18h] [rbp-88h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v45; // [rsp+28h] [rbp-78h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  __int64 v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+30h] [rbp-70h]
  int v49; // [rsp+38h] [rbp-68h]
  __int64 v50; // [rsp+48h] [rbp-58h] BYREF
  __int64 v51; // [rsp+50h] [rbp-50h] BYREF
  __int64 v52; // [rsp+58h] [rbp-48h] BYREF
  __int64 v53; // [rsp+60h] [rbp-40h] BYREF
  __int64 v54[7]; // [rsp+68h] [rbp-38h] BYREF

  v7 = a4;
  v10 = a6;
  if ( a6 )
    goto LABEL_19;
  v12 = *a1;
  v51 = a2;
  v52 = a3;
  v53 = a4;
  v54[0] = a5;
  v13 = *(_DWORD *)(v12 + 1584);
  v48 = *(_QWORD *)(v12 + 1568);
  if ( v13 )
  {
    if ( a2 && *(_BYTE *)a2 == 1 )
    {
      v14 = *(_QWORD *)(a2 + 136);
      v15 = *(_DWORD *)(v14 + 32);
      v16 = *(__int64 **)(v14 + 24);
      if ( v15 > 0x40 )
      {
        v17 = *v16;
      }
      else
      {
        v17 = 0;
        if ( v15 )
          v17 = (__int64)((_QWORD)v16 << (64 - (unsigned __int8)v15)) >> (64 - (unsigned __int8)v15);
      }
      v40 = a5;
      v45 = v12;
      v50 = v17;
      v18 = sub_AF7D50(&v50, &v52, &v53, v54);
      v19 = v45;
      v7 = a4;
      v20 = v40;
    }
    else
    {
      v41 = a5;
      v47 = v12;
      v18 = sub_AF81D0(&v51, &v52, &v53, v54);
      v20 = v41;
      v7 = a4;
      v19 = v47;
    }
    v21 = v13 - 1;
    v43 = v19;
    v22 = 1;
    v23 = 0;
    v46 = v20;
    for ( i = (v13 - 1) & v18; ; i = v21 & v26 )
    {
      v27 = *(_QWORD *)(v48 + 8LL * i);
      if ( v27 == -4096 )
      {
        a5 = v46;
        v10 = v23;
        goto LABEL_18;
      }
      if ( v27 != -8192 )
      {
        v28 = *(_BYTE *)(v27 - 16);
        v25 = (v28 & 2) != 0 ? *(_QWORD **)(v27 - 32) : (_QWORD *)(v27 - 16 - 8LL * ((v28 >> 2) & 0xF));
        if ( v51 == *v25 )
        {
          v39 = v23;
          v37 = v7;
          v38 = v22;
          v49 = v21;
          v34 = sub_A17150((_BYTE *)(v27 - 16));
          v21 = v49;
          v22 = v38;
          v7 = v37;
          v23 = v39;
          if ( v52 == *((_QWORD *)v34 + 1) )
          {
            v35 = sub_A17150((_BYTE *)(v27 - 16));
            v21 = v49;
            v22 = v38;
            v7 = v37;
            v23 = v39;
            if ( v53 == *((_QWORD *)v35 + 2) )
            {
              v36 = sub_A17150((_BYTE *)(v27 - 16));
              v21 = v49;
              v22 = v38;
              v7 = v37;
              v23 = v39;
              if ( v54[0] == *((_QWORD *)v36 + 3) )
                break;
            }
          }
        }
      }
      v26 = v22 + i;
      ++v22;
    }
    v10 = v39;
    a5 = v46;
    if ( v48 + 8LL * i != *(_QWORD *)(v43 + 1568) + 8LL * *(unsigned int *)(v43 + 1584) )
      return v27;
  }
LABEL_18:
  result = 0;
  if ( a7 )
  {
LABEL_19:
    v30 = *a1;
    v51 = a2;
    v52 = a3;
    v53 = v7;
    v31 = v30 + 1560;
    v54[0] = a5;
    v32 = sub_B97910(16, 4, v10);
    v33 = v32;
    if ( v32 )
      sub_AF2980(v32, (int)a1, v10, (int)&v51, 4);
    return sub_B03610(v33, v10, v31);
  }
  return result;
}

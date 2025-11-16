// Function: sub_3205010
// Address: 0x3205010
//
__int64 __fastcall sub_3205010(__int64 a1, __int64 a2)
{
  unsigned __int64 *v3; // rbx
  __int64 v4; // rbx
  unsigned __int8 v5; // al
  unsigned __int16 v6; // ax
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned __int64 *v10; // rdx
  unsigned int v11; // r13d
  unsigned int v12; // esi
  __int64 v13; // rdi
  int v14; // r11d
  __int64 v15; // r10
  unsigned int v16; // ecx
  __int64 v17; // rax
  unsigned __int64 *v18; // r9
  int v19; // eax
  __int64 v20; // rdx
  int v21; // eax
  int v22; // ecx
  unsigned __int64 *v23; // rdi
  unsigned __int16 v24; // ax
  int v25; // eax
  int v26; // edi
  int v27; // r10d
  _QWORD *v28; // r9
  __int64 v29; // rsi
  unsigned int v30; // edx
  _QWORD *v31; // rax
  unsigned __int64 *v32; // r8
  unsigned int *v33; // rax
  _QWORD *v34; // rdx
  unsigned __int64 *v35; // rax
  unsigned __int64 *v36; // [rsp+10h] [rbp-40h] BYREF
  __int64 v37; // [rsp+18h] [rbp-38h] BYREF
  unsigned __int64 *v38; // [rsp+20h] [rbp-30h] BYREF
  int v39; // [rsp+28h] [rbp-28h]

  v3 = (unsigned __int64 *)a2;
  if ( (unsigned __int16)sub_AF18C0(a2) == 22 )
    sub_3206530(a1, a2, 0);
  while ( (unsigned __int16)sub_AF18C0((__int64)v3) == 22 )
  {
    v5 = *((_BYTE *)v3 - 16);
    if ( (v5 & 2) != 0 )
      v4 = *(v3 - 4);
    else
      v4 = (__int64)&v3[-((v5 >> 2) & 0xF) - 2];
    v3 = *(unsigned __int64 **)(v4 + 24);
  }
  v6 = sub_AF18C0((__int64)v3);
  if ( v6 > 0x17u || ((1LL << v6) & 0x880004) == 0 )
    return sub_3206530(a1, v3, 0);
  v36 = v3;
  ++*(_DWORD *)(a1 + 1328);
  sub_A547D0((__int64)v3, 2);
  if ( !v8 )
  {
    sub_A547D0((__int64)v36, 7);
    if ( !v20 )
    {
      v10 = v36;
LABEL_13:
      v12 = *(_DWORD *)(a1 + 1272);
      v38 = v10;
      v39 = 0;
      if ( v12 )
      {
        v13 = *(_QWORD *)(a1 + 1256);
        v14 = 1;
        v15 = 0;
        v16 = (v12 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v17 = v13 + 16LL * v16;
        v18 = *(unsigned __int64 **)v17;
        if ( *(unsigned __int64 **)v17 == v10 )
        {
LABEL_15:
          v11 = *(_DWORD *)(v17 + 8);
          goto LABEL_16;
        }
        while ( v18 != (unsigned __int64 *)-4096LL )
        {
          if ( v18 == (unsigned __int64 *)-8192LL && !v15 )
            v15 = v17;
          v16 = (v12 - 1) & (v14 + v16);
          v17 = v13 + 16LL * v16;
          v18 = *(unsigned __int64 **)v17;
          if ( *(unsigned __int64 **)v17 == v10 )
            goto LABEL_15;
          ++v14;
        }
        if ( !v15 )
          v15 = v17;
        v21 = *(_DWORD *)(a1 + 1264);
        ++*(_QWORD *)(a1 + 1248);
        v22 = v21 + 1;
        v37 = v15;
        if ( 4 * (v21 + 1) < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 1268) - v22 > v12 >> 3 )
            goto LABEL_31;
          goto LABEL_48;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 1248);
        v37 = 0;
      }
      v12 *= 2;
LABEL_48:
      sub_31FEE60(a1 + 1248, v12);
      sub_31FC0C0(a1 + 1248, (__int64 *)&v38, &v37);
      v10 = v38;
      v15 = v37;
      v22 = *(_DWORD *)(a1 + 1264) + 1;
LABEL_31:
      *(_DWORD *)(a1 + 1264) = v22;
      if ( *(_QWORD *)v15 != -4096 )
        --*(_DWORD *)(a1 + 1268);
      *(_QWORD *)v15 = v10;
      v23 = v36;
      *(_DWORD *)(v15 + 8) = v39;
      v24 = sub_AF18C0((__int64)v23);
      if ( v24 == 23 )
      {
        v11 = sub_3204E90((_QWORD *)a1, v36);
      }
      else
      {
        if ( v24 > 0x17u || v24 != 2 && v24 != 19 )
          BUG();
        v11 = sub_32057A0(a1, v36);
      }
      v25 = *(_DWORD *)(a1 + 1272);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = 1;
        v28 = 0;
        v29 = *(_QWORD *)(a1 + 1256);
        v30 = (v25 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
        v31 = (_QWORD *)(v29 + 16LL * v30);
        v32 = (unsigned __int64 *)*v31;
        if ( v36 == (unsigned __int64 *)*v31 )
        {
LABEL_37:
          v33 = (unsigned int *)(v31 + 1);
LABEL_38:
          *v33 = v11;
          goto LABEL_16;
        }
        while ( v32 != (unsigned __int64 *)-4096LL )
        {
          if ( v32 == (unsigned __int64 *)-8192LL && !v28 )
            v28 = v31;
          v30 = v26 & (v27 + v30);
          v31 = (_QWORD *)(v29 + 16LL * v30);
          v32 = (unsigned __int64 *)*v31;
          if ( v36 == (unsigned __int64 *)*v31 )
            goto LABEL_37;
          ++v27;
        }
        if ( !v28 )
          v28 = v31;
      }
      else
      {
        v28 = 0;
      }
      v34 = sub_31FF040(a1 + 1248, &v36, v28);
      v35 = v36;
      *((_DWORD *)v34 + 2) = 0;
      *v34 = v35;
      v33 = (unsigned int *)(v34 + 1);
      goto LABEL_38;
    }
  }
  v9 = sub_3206530(a1, v36, 0);
  v10 = v36;
  v11 = v9;
  if ( (*((_BYTE *)v36 + 20) & 4) == 0 )
    goto LABEL_13;
LABEL_16:
  v19 = *(_DWORD *)(a1 + 1328);
  if ( v19 == 1 )
  {
    sub_32053F0(a1);
    v19 = *(_DWORD *)(a1 + 1328);
  }
  *(_DWORD *)(a1 + 1328) = v19 - 1;
  return v11;
}

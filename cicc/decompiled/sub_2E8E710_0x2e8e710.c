// Function: sub_2E8E710
// Address: 0x2e8e710
//
__int64 __fastcall sub_2E8E710(__int64 a1, unsigned int a2, __int64 a3, char a4, char a5)
{
  unsigned int v5; // r11d
  int v7; // r14d
  __int64 v9; // rbx
  unsigned int v10; // edx
  __int64 v11; // r10
  __int64 v12; // rcx
  char v13; // r15
  __int64 v14; // r12
  char v15; // al
  unsigned int v16; // esi
  char v17; // al
  int v19; // eax
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int16 *v23; // rax
  __int16 *v24; // rax
  __int16 v25; // di
  int v26; // esi
  __int64 v27; // [rsp+0h] [rbp-A0h]
  __int64 v28; // [rsp+8h] [rbp-98h]
  unsigned int v29; // [rsp+10h] [rbp-90h]
  __int64 v30; // [rsp+10h] [rbp-90h]
  __int64 v31; // [rsp+18h] [rbp-88h]
  unsigned int v32; // [rsp+18h] [rbp-88h]
  __int64 v33; // [rsp+20h] [rbp-80h]
  unsigned int v34; // [rsp+28h] [rbp-78h]
  int v36; // [rsp+3Ch] [rbp-64h] BYREF
  int v37; // [rsp+40h] [rbp-60h] BYREF
  __int16 *v38; // [rsp+48h] [rbp-58h]
  __int16 v39; // [rsp+50h] [rbp-50h]
  int v40; // [rsp+58h] [rbp-48h]
  __int64 v41; // [rsp+60h] [rbp-40h]
  __int16 v42; // [rsp+68h] [rbp-38h]

  v5 = a2 - 1;
  v7 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( !v7 )
    return 0xFFFFFFFFLL;
  v9 = 0;
  v10 = a2;
  v11 = 4LL * (a2 >> 5);
  v12 = a1;
  v13 = (v5 <= 0x3FFFFFFE) & a5;
  v33 = 24LL * a2;
  while ( 1 )
  {
    v14 = *(_QWORD *)(v12 + 32) + 40 * v9;
    v15 = *(_BYTE *)v14;
    if ( v13 && v15 == 12 )
    {
      v19 = *(_DWORD *)(*(_QWORD *)(v14 + 24) + v11);
      if ( !_bittest(&v19, v10) )
        return (unsigned int)v9;
      goto LABEL_16;
    }
    if ( v15 || (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
      goto LABEL_16;
    v16 = *(_DWORD *)(v14 + 8);
    if ( v10 == v16 )
      goto LABEL_14;
    if ( a3 )
      break;
    if ( v10 == v16 )
      goto LABEL_14;
LABEL_16:
    if ( v7 == (_DWORD)++v9 )
      return 0xFFFFFFFFLL;
  }
  if ( v5 > 0x3FFFFFFE || v16 - 1 > 0x3FFFFFFE )
    goto LABEL_16;
  v27 = v12;
  v28 = v11;
  v34 = v5;
  if ( a5 )
  {
    v29 = v10;
    v31 = a3;
    v17 = sub_E92070(a3, v16, v10);
    a3 = v31;
    v5 = v34;
    v10 = v29;
    v11 = v28;
    v12 = v27;
  }
  else
  {
    v20 = *(_QWORD *)(a3 + 8);
    v36 = *(_DWORD *)(v14 + 8);
    v30 = a3;
    v21 = *(unsigned int *)(v20 + v33 + 8);
    v22 = *(_QWORD *)(a3 + 56);
    v32 = v10;
    v40 = 0;
    v23 = (__int16 *)(v22 + 2 * v21);
    v41 = 0;
    LODWORD(v21) = *v23;
    v24 = v23 + 1;
    v25 = v21;
    v26 = v10 + v21;
    v37 = v26;
    if ( !v25 )
      v24 = 0;
    v39 = v26;
    v38 = v24;
    v42 = 0;
    v17 = sub_2E46590(&v37, &v36);
    v12 = v27;
    v5 = v34;
    v11 = v28;
    a3 = v30;
    v10 = v32;
  }
  if ( !v17 )
    goto LABEL_16;
LABEL_14:
  if ( a4 && (((*(_BYTE *)(v14 + 3) & 0x10) != 0) & (*(_BYTE *)(v14 + 3) >> 6)) == 0 )
    goto LABEL_16;
  return (unsigned int)v9;
}

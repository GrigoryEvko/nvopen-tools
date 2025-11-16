// Function: sub_2A689D0
// Address: 0x2a689d0
//
void __fastcall sub_2A689D0(__int64 a1, __int64 a2, unsigned __int8 *a3, __int64 a4)
{
  unsigned int v5; // r8d
  int v7; // r11d
  __int64 v8; // rcx
  __int64 *v9; // r9
  unsigned int v10; // edx
  _QWORD *v11; // rsi
  __int64 v12; // rax
  _BYTE *v13; // rsi
  int v14; // eax
  int v15; // edx
  __int64 v16; // rcx
  int v17; // esi
  __int64 v18; // [rsp+8h] [rbp-78h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-68h] BYREF
  char v20[8]; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-58h]
  unsigned int v22; // [rsp+30h] [rbp-50h]
  unsigned __int64 v23; // [rsp+38h] [rbp-48h]
  unsigned int v24; // [rsp+40h] [rbp-40h]

  v18 = a2;
  sub_22C05A0((__int64)v20, a3);
  v5 = *(_DWORD *)(a1 + 160);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 136);
    v19 = 0;
LABEL_26:
    v17 = 2 * v5;
LABEL_27:
    sub_2A68410(a1 + 136, v17);
    sub_2A65730(a1 + 136, &v18, &v19);
    v16 = v18;
    v9 = v19;
    v15 = *(_DWORD *)(a1 + 152) + 1;
    goto LABEL_22;
  }
  v7 = 1;
  v8 = *(_QWORD *)(a1 + 144);
  v9 = 0;
  v10 = (v5 - 1) & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
  v11 = (_QWORD *)(v8 + 48LL * v10);
  v12 = *v11;
  if ( v18 == *v11 )
  {
LABEL_3:
    v13 = v11 + 1;
    goto LABEL_4;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = (v5 - 1) & (v7 + v10);
    v11 = (_QWORD *)(v8 + 48LL * v10);
    v12 = *v11;
    if ( v18 == *v11 )
      goto LABEL_3;
    ++v7;
  }
  v14 = *(_DWORD *)(a1 + 152);
  if ( !v9 )
    v9 = v11;
  ++*(_QWORD *)(a1 + 136);
  v15 = v14 + 1;
  v19 = v9;
  if ( 4 * (v14 + 1) >= 3 * v5 )
    goto LABEL_26;
  v16 = a2;
  if ( v5 - *(_DWORD *)(a1 + 156) - v15 <= v5 >> 3 )
  {
    v17 = v5;
    goto LABEL_27;
  }
LABEL_22:
  *(_DWORD *)(a1 + 152) = v15;
  if ( *v9 != -4096 )
    --*(_DWORD *)(a1 + 156);
  *v9 = v16;
  v13 = v9 + 1;
  *((_WORD *)v9 + 4) = 0;
LABEL_4:
  sub_2A639B0(a1, v13, a2, (__int64)v20, a4);
  if ( (unsigned int)(unsigned __int8)v20[0] - 4 <= 1 )
  {
    if ( v24 > 0x40 && v23 )
      j_j___libc_free_0_0(v23);
    if ( v22 > 0x40 )
    {
      if ( v21 )
        j_j___libc_free_0_0(v21);
    }
  }
}

// Function: sub_2A68BC0
// Address: 0x2a68bc0
//
_QWORD *__fastcall sub_2A68BC0(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // r15
  unsigned int v4; // esi
  __int64 v5; // rcx
  int v6; // r8d
  __int64 *v7; // r14
  unsigned int v8; // edx
  _QWORD *v9; // r12
  unsigned __int8 *v10; // rax
  _QWORD *v11; // r12
  int v13; // eax
  int v14; // edx
  __int64 v15; // rcx
  unsigned __int8 v16; // al
  __int64 *v17; // [rsp+8h] [rbp-98h] BYREF
  unsigned __int8 v18[48]; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int8 *v19; // [rsp+40h] [rbp-60h] BYREF
  __int64 v20; // [rsp+48h] [rbp-58h]
  __int64 v21; // [rsp+50h] [rbp-50h]
  int v22; // [rsp+58h] [rbp-48h]
  __int64 v23; // [rsp+60h] [rbp-40h]
  int v24; // [rsp+68h] [rbp-38h]

  v2 = a1 + 136;
  v19 = a2;
  v4 = *(_DWORD *)(a1 + 160);
  *(_QWORD *)v18 = 0;
  v20 = 0;
  if ( !v4 )
  {
    ++*(_QWORD *)(a1 + 136);
    v17 = 0;
LABEL_25:
    v4 *= 2;
    goto LABEL_26;
  }
  v5 = *(_QWORD *)(a1 + 144);
  v6 = 1;
  v7 = 0;
  v8 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v9 = (_QWORD *)(v5 + 48LL * v8);
  v10 = (unsigned __int8 *)*v9;
  if ( a2 == (unsigned __int8 *)*v9 )
  {
LABEL_3:
    v11 = v9 + 1;
    sub_22C0090(v18);
    return v11;
  }
  while ( v10 != (unsigned __int8 *)-4096LL )
  {
    if ( v10 == (unsigned __int8 *)-8192LL && !v7 )
      v7 = v9;
    v8 = (v4 - 1) & (v6 + v8);
    v9 = (_QWORD *)(v5 + 48LL * v8);
    v10 = (unsigned __int8 *)*v9;
    if ( a2 == (unsigned __int8 *)*v9 )
      goto LABEL_3;
    ++v6;
  }
  v13 = *(_DWORD *)(a1 + 152);
  if ( !v7 )
    v7 = v9;
  ++*(_QWORD *)(a1 + 136);
  v14 = v13 + 1;
  v17 = v7;
  if ( 4 * (v13 + 1) >= 3 * v4 )
    goto LABEL_25;
  v15 = (__int64)a2;
  if ( v4 - *(_DWORD *)(a1 + 156) - v14 <= v4 >> 3 )
  {
LABEL_26:
    sub_2A68410(v2, v4);
    sub_2A65730(v2, (__int64 *)&v19, &v17);
    v15 = (__int64)v19;
    v7 = v17;
    v14 = *(_DWORD *)(a1 + 152) + 1;
  }
  *(_DWORD *)(a1 + 152) = v14;
  if ( *v7 != -4096 )
    --*(_DWORD *)(a1 + 156);
  *v7 = v15;
  v16 = v20;
  *((_WORD *)v7 + 4) = (unsigned __int8)v20;
  if ( v16 > 3u )
  {
    if ( (unsigned __int8)(v16 - 4) <= 1u )
    {
      *((_DWORD *)v7 + 6) = v22;
      v7[2] = v21;
      *((_DWORD *)v7 + 10) = v24;
      v7[4] = v23;
      *((_BYTE *)v7 + 9) = BYTE1(v20);
    }
  }
  else if ( v16 > 1u )
  {
    v7[2] = v21;
  }
  v11 = v7 + 1;
  sub_22C0090(v18);
  if ( *a2 <= 0x15u )
    sub_2A624B0((__int64)(v7 + 1), a2, 0);
  return v11;
}

// Function: sub_2471F40
// Address: 0x2471f40
//
__int64 __fastcall sub_2471F40(__int64 *a1, unsigned __int8 *a2, unsigned int **a3)
{
  int v3; // r13d
  __int64 v4; // rax
  __int64 v5; // rdx
  unsigned int v6; // eax
  int v7; // edx
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r15
  int v12; // r15d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rbx
  unsigned int v16; // r12d
  unsigned __int64 v17; // r15
  unsigned int v18; // r8d
  unsigned __int64 v19; // r10
  bool v20; // zf
  unsigned int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  __int64 v24; // rdx
  unsigned __int64 v25; // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  unsigned __int64 v29; // [rsp+0h] [rbp-80h]
  unsigned int v30; // [rsp+10h] [rbp-70h]
  unsigned int v31; // [rsp+10h] [rbp-70h]
  __int64 v32; // [rsp+10h] [rbp-70h]
  unsigned int v33; // [rsp+1Ch] [rbp-64h]
  __int64 *v35; // [rsp+30h] [rbp-50h]
  _BYTE *v36; // [rsp+38h] [rbp-48h]
  unsigned __int64 v37; // [rsp+40h] [rbp-40h] BYREF
  __int64 v38; // [rsp+48h] [rbp-38h]

  v36 = (_BYTE *)sub_B2BEC0(a1[1]);
  v4 = sub_9208B0((__int64)v36, *(_QWORD *)(a1[2] + 80));
  v38 = v5;
  v37 = (unsigned __int64)(v4 + 7) >> 3;
  v6 = sub_CA1930(&v37);
  v7 = *a2;
  v30 = v6;
  v33 = v6;
  if ( v7 == 40 )
  {
    v8 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)a2);
  }
  else
  {
    v8 = -32;
    if ( v7 != 85 )
    {
      v8 = -96;
      if ( v7 != 34 )
        BUG();
    }
  }
  if ( (a2[7] & 0x80u) == 0 )
    goto LABEL_9;
  v9 = sub_BD2BC0((__int64)a2);
  v11 = v9 + v10;
  if ( (a2[7] & 0x80u) == 0 )
  {
    if ( (unsigned int)(v11 >> 4) )
      goto LABEL_28;
  }
  else if ( (unsigned int)((v11 - sub_BD2BC0((__int64)a2)) >> 4) )
  {
    if ( (a2[7] & 0x80u) != 0 )
    {
      v12 = *(_DWORD *)(sub_BD2BC0((__int64)a2) + 8);
      if ( (a2[7] & 0x80u) == 0 )
        BUG();
      v13 = sub_BD2BC0((__int64)a2);
      v8 -= 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
      goto LABEL_9;
    }
LABEL_28:
    BUG();
  }
LABEL_9:
  v35 = (__int64 *)&a2[v8];
  v15 = (__int64 *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( v15 == v35 )
  {
    v26 = 0;
    goto LABEL_21;
  }
  v29 = v30;
  v16 = 0;
  v17 = 0;
  do
  {
    while ( (unsigned int)(*(_DWORD *)(*((_QWORD *)a2 + 10) + 12LL) - 1) > v17 )
    {
LABEL_15:
      v15 += 4;
      ++v17;
      if ( v15 == v35 )
        goto LABEL_20;
    }
    v37 = sub_BDB740((__int64)v36, *(_QWORD *)(*v15 + 8));
    v38 = v24;
    v25 = sub_CA1930(&v37);
    if ( *v36 && v29 > v25 )
    {
      v18 = v33 + v16;
      v16 = v33 + v16 - v25;
      if ( v18 <= 0x320 )
        goto LABEL_13;
    }
    else
    {
      v18 = v16 + v25;
      if ( v16 + (unsigned int)v25 <= 0x320 )
      {
LABEL_13:
        v31 = v18;
        v19 = sub_2464620((__int64)a1, a3, v16);
        v20 = v31 == 0;
        v21 = v31;
        v32 = v19;
        v16 = v33 * ((v21 - !v20) / v33 + !v20);
        if ( v19 )
        {
          LOBYTE(v3) = byte_4FE8EA8;
          v22 = v3;
          BYTE1(v22) = 1;
          v3 = v22;
          v23 = sub_246F3F0(a1[3], *v15);
          sub_2463EC0((__int64 *)a3, v23, v32, v3, 0);
        }
        goto LABEL_15;
      }
    }
    v15 += 4;
    ++v17;
    v16 = v33 * ((v18 - 1) / v33 + 1);
  }
  while ( v15 != v35 );
LABEL_20:
  v26 = v16;
LABEL_21:
  v27 = sub_AD64C0(*(_QWORD *)(a1[2] + 80), v26, 0);
  return sub_2463EC0((__int64 *)a3, v27, *(_QWORD *)(a1[2] + 152), 0, 0);
}

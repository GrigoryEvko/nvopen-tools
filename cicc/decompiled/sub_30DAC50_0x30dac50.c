// Function: sub_30DAC50
// Address: 0x30dac50
//
__int64 __fastcall sub_30DAC50(__int64 a1, __int64 a2)
{
  unsigned int v4; // eax
  __int64 v5; // rdx
  bool v6; // zf
  int v7; // r14d
  unsigned int v8; // r14d
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rsi
  _QWORD *v12; // rax
  _QWORD *v13; // rdx
  __int64 *v14; // rax
  bool v15; // cc
  __int64 v16; // r14
  unsigned int v18; // esi
  __int64 v19; // rcx
  int v20; // r11d
  __int64 *v21; // rdi
  __int64 v22; // r9
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rax
  _BYTE *v28; // rdx
  bool v29; // r8
  __int64 *v30; // rax
  __int64 v31; // rcx
  __int64 v32; // rdx
  int v33; // eax
  int v34; // edx
  __int64 v35; // rdx
  __int64 *v36; // [rsp+0h] [rbp-E0h]
  __int64 *v37; // [rsp+0h] [rbp-E0h]
  _BYTE *v38; // [rsp+8h] [rbp-D8h]
  _BYTE *v39; // [rsp+8h] [rbp-D8h]
  _BYTE *v40; // [rsp+8h] [rbp-D8h]
  __int64 v41; // [rsp+10h] [rbp-D0h]
  bool v42; // [rsp+1Eh] [rbp-C2h]
  bool v43; // [rsp+1Fh] [rbp-C1h]
  __int64 v44; // [rsp+20h] [rbp-C0h]
  _BYTE *v45; // [rsp+20h] [rbp-C0h]
  __int64 v46; // [rsp+28h] [rbp-B8h]
  _BYTE *v47; // [rsp+28h] [rbp-B8h]
  __int64 v48; // [rsp+38h] [rbp-A8h] BYREF
  const void *v49; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v50; // [rsp+48h] [rbp-98h]
  __int64 v51; // [rsp+50h] [rbp-90h]
  const void *v52; // [rsp+58h] [rbp-88h] BYREF
  unsigned int v53; // [rsp+60h] [rbp-80h]
  __int64 v54; // [rsp+70h] [rbp-70h]
  const void *v55; // [rsp+78h] [rbp-68h] BYREF
  unsigned int v56; // [rsp+80h] [rbp-60h]
  __int64 v57[2]; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v58; // [rsp+A0h] [rbp-40h]

  v4 = sub_AE2980(*(_QWORD *)(a1 + 80), 0)[1];
  v50 = v4;
  if ( v4 <= 0x40 )
  {
    v49 = 0;
    v5 = *(_QWORD *)(a2 + 8);
    v51 = 0;
    v6 = *(_BYTE *)(v5 + 8) == 14;
    v53 = v4;
    v43 = v6;
    goto LABEL_3;
  }
  sub_C43690((__int64)&v49, 0, 0);
  v32 = *(_QWORD *)(a2 + 8);
  v51 = 0;
  v6 = *(_BYTE *)(v32 + 8) == 14;
  v53 = v50;
  v43 = v6;
  if ( v50 <= 0x40 )
  {
LABEL_3:
    v7 = *(_DWORD *)(a2 + 4);
    v52 = v49;
    v8 = v7 & 0x7FFFFFF;
    if ( !v8 )
      goto LABEL_18;
    goto LABEL_4;
  }
  sub_C43780((__int64)&v52, &v49);
  v8 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  if ( v8 )
  {
LABEL_4:
    v9 = 0;
    v44 = 0;
    v10 = 8LL * v8;
    v46 = 0;
    v41 = a1 + 424;
    while ( 1 )
    {
      v6 = *(_BYTE *)(a1 + 292) == 0;
      v11 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v9);
      v48 = v11;
      if ( v6 )
      {
        if ( !sub_C8CA60(a1 + 264, v11) )
          goto LABEL_26;
LABEL_10:
        v9 += 8;
        if ( v9 == v10 )
          goto LABEL_11;
      }
      else
      {
        v12 = *(_QWORD **)(a1 + 272);
        v13 = &v12[*(unsigned int *)(a1 + 284)];
        if ( v12 != v13 )
        {
          while ( v11 != *v12 )
          {
            if ( v13 == ++v12 )
              goto LABEL_26;
          }
          goto LABEL_10;
        }
LABEL_26:
        v18 = *(_DWORD *)(a1 + 448);
        if ( !v18 )
        {
          ++*(_QWORD *)(a1 + 424);
          v57[0] = 0;
          goto LABEL_76;
        }
        v19 = v48;
        v20 = 1;
        v21 = 0;
        v22 = *(_QWORD *)(a1 + 432);
        v23 = (v18 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
        v24 = (__int64 *)(v22 + 16LL * v23);
        v25 = *v24;
        if ( v48 != *v24 )
        {
          while ( v25 != -4096 )
          {
            if ( v25 == -8192 && !v21 )
              v21 = v24;
            v23 = (v18 - 1) & (v20 + v23);
            v24 = (__int64 *)(v22 + 16LL * v23);
            v25 = *v24;
            if ( v48 == *v24 )
              goto LABEL_28;
            ++v20;
          }
          if ( !v21 )
            v21 = v24;
          v33 = *(_DWORD *)(a1 + 440);
          ++*(_QWORD *)(a1 + 424);
          v34 = v33 + 1;
          v57[0] = (__int64)v21;
          if ( 4 * (v33 + 1) < 3 * v18 )
          {
            if ( v18 - *(_DWORD *)(a1 + 444) - v34 > v18 >> 3 )
            {
LABEL_72:
              *(_DWORD *)(a1 + 440) = v34;
              if ( *v21 != -4096 )
                --*(_DWORD *)(a1 + 444);
              *v21 = v19;
              v21[1] = 0;
              goto LABEL_30;
            }
LABEL_77:
            sub_22E02D0(v41, v18);
            sub_27EFA30(v41, &v48, v57);
            v19 = v48;
            v21 = (__int64 *)v57[0];
            v34 = *(_DWORD *)(a1 + 440) + 1;
            goto LABEL_72;
          }
LABEL_76:
          v18 *= 2;
          goto LABEL_77;
        }
LABEL_28:
        v26 = v24[1];
        if ( v26 && *(_QWORD *)(a2 + 40) != v26 )
          goto LABEL_10;
LABEL_30:
        v27 = *(_QWORD *)(a2 - 8);
        v28 = *(_BYTE **)(v27 + 4 * v9);
        if ( !v28 )
          BUG();
        if ( (_BYTE *)a2 == v28 )
          goto LABEL_10;
        v29 = 0;
        v30 = *(__int64 **)(v27 + 4 * v9);
        if ( *v28 > 0x15u )
        {
          v57[0] = (__int64)v28;
          v40 = v28;
          v30 = sub_30D7570(a1 + 136, v57);
          v29 = v43;
          v28 = v40;
          if ( v30 )
          {
            v30 = (__int64 *)v30[1];
            v29 = v43 && v30 == 0;
          }
        }
        v54 = 0;
        v56 = v50;
        if ( v50 > 0x40 )
        {
          v42 = v29;
          v37 = v30;
          v39 = v28;
          sub_C43780((__int64)&v55, &v49);
          v29 = v42;
          v30 = v37;
          v28 = v39;
        }
        else
        {
          v55 = v49;
        }
        if ( v29 )
        {
          v36 = v30;
          v38 = v28;
          sub_30D74B0((__int64)v57, a1 + 232, (__int64)v28);
          v31 = v57[0];
          v28 = v38;
          v30 = v36;
          v54 = v57[0];
          if ( v56 > 0x40 && v55 )
          {
            j_j___libc_free_0_0((unsigned __int64)v55);
            v31 = v54;
            v30 = v36;
            v28 = v38;
          }
          v55 = (const void *)v57[1];
          v56 = v58;
        }
        else
        {
          if ( v30 )
          {
            if ( v46 )
              goto LABEL_38;
            goto LABEL_48;
          }
          v31 = v54;
        }
        if ( !v31 )
          goto LABEL_39;
        if ( v46 )
        {
LABEL_38:
          if ( v30 != (__int64 *)v46 )
            goto LABEL_39;
          goto LABEL_52;
        }
LABEL_48:
        if ( v44 )
        {
          if ( v51 != v54 )
            goto LABEL_39;
          if ( v53 <= 0x40 )
          {
            if ( v52 != v55 )
            {
LABEL_39:
              if ( v56 > 0x40 && v55 )
                j_j___libc_free_0_0((unsigned __int64)v55);
              goto LABEL_18;
            }
          }
          else if ( !sub_C43C50((__int64)&v52, &v55) )
          {
            goto LABEL_39;
          }
        }
        else
        {
          v46 = (__int64)v30;
          if ( !v30 )
          {
            v51 = v54;
            if ( v53 <= 0x40 && v56 <= 0x40 )
            {
              v53 = v56;
              v52 = v55;
            }
            else
            {
              v45 = v28;
              sub_C43990((__int64)&v52, (__int64)&v55);
              v28 = v45;
            }
            v47 = v28;
            sub_969240((__int64 *)&v55);
            v35 = (__int64)v47;
            v46 = 0;
            v44 = v35;
            goto LABEL_10;
          }
        }
LABEL_52:
        if ( v56 <= 0x40 || !v55 )
          goto LABEL_10;
        j_j___libc_free_0_0((unsigned __int64)v55);
        v9 += 8;
        if ( v9 == v10 )
        {
LABEL_11:
          if ( !v46 )
            goto LABEL_12;
          v57[0] = a2;
          *sub_30D9190(a1 + 136, v57) = v46;
          goto LABEL_18;
        }
      }
    }
  }
  v44 = 0;
LABEL_12:
  if ( v51 )
  {
    v57[0] = a2;
    v14 = sub_30DA4E0(a1 + 232, v57);
    v15 = *((_DWORD *)v14 + 4) <= 0x40u;
    *v14 = v51;
    if ( v15 && v53 <= 0x40 )
    {
      v14[1] = (__int64)v52;
      *((_DWORD *)v14 + 4) = v53;
    }
    else
    {
      sub_C43990((__int64)(v14 + 1), (__int64)&v52);
    }
    v16 = sub_30D1740(a1, v44);
    if ( v16 )
    {
      v57[0] = a2;
      *sub_30DA630(a1 + 168, v57) = v16;
    }
  }
LABEL_18:
  if ( v53 > 0x40 && v52 )
    j_j___libc_free_0_0((unsigned __int64)v52);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0((unsigned __int64)v49);
  return 1;
}

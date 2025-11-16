// Function: sub_13C4F40
// Address: 0x13c4f40
//
__int64 __fastcall sub_13C4F40(__int64 a1, _QWORD *a2)
{
  __int64 v4; // rdi
  __int64 v5; // r15
  __int64 v6; // r13
  unsigned __int8 v7; // al
  unsigned int v8; // r12d
  _QWORD *v10; // rax
  __int64 v11; // r15
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rsi
  _BYTE *v19; // rsi
  _BYTE *v20; // r13
  __int64 v21; // rcx
  __int64 v22; // rdi
  unsigned int v23; // edx
  _QWORD *v24; // rax
  __int64 v25; // r9
  _BYTE *v26; // r14
  __int64 v27; // r15
  _QWORD *v28; // r13
  __int64 v29; // rax
  __int64 v30; // rax
  unsigned int v31; // esi
  int v32; // eax
  __int64 v33; // rsi
  int v34; // r8d
  __int64 v35; // rdi
  int v36; // ecx
  unsigned int v37; // edx
  __int64 v38; // r9
  __int64 v39; // rdx
  int v40; // r11d
  _QWORD *v41; // r10
  int v42; // ecx
  int v43; // eax
  __int64 v44; // rdi
  int v45; // esi
  int v46; // r11d
  _QWORD *v47; // r10
  __int64 v48; // r8
  unsigned int v49; // edx
  __int64 v50; // r9
  _QWORD *v51; // rsi
  unsigned int v52; // edi
  _QWORD *v53; // rcx
  int v54; // r11d
  __int64 v55; // [rsp+8h] [rbp-68h]
  _QWORD *v56; // [rsp+18h] [rbp-58h] BYREF
  _BYTE *v57; // [rsp+20h] [rbp-50h] BYREF
  _BYTE *v58; // [rsp+28h] [rbp-48h]
  _BYTE *v59; // [rsp+30h] [rbp-40h]

  v4 = *(a2 - 3);
  v57 = 0;
  v58 = 0;
  v59 = 0;
  if ( v4 )
  {
    if ( (unsigned __int8)sub_1593BB0(v4) )
    {
      v5 = a2[1];
      if ( v5 )
        goto LABEL_4;
LABEL_28:
      v20 = v58;
      if ( v58 != v57 )
      {
        v55 = a1 + 232;
        while ( 1 )
        {
          v31 = *(_DWORD *)(a1 + 256);
          if ( !v31 )
            break;
          v21 = *((_QWORD *)v20 - 1);
          v22 = *(_QWORD *)(a1 + 240);
          v23 = (v31 - 1) & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
          v24 = (_QWORD *)(v22 + 16LL * v23);
          v25 = *v24;
          if ( v21 != *v24 )
          {
            v40 = 1;
            v41 = 0;
            while ( v25 != -8 )
            {
              if ( !v41 && v25 == -16 )
                v41 = v24;
              v23 = (v31 - 1) & (v40 + v23);
              v24 = (_QWORD *)(v22 + 16LL * v23);
              v25 = *v24;
              if ( v21 == *v24 )
                goto LABEL_31;
              ++v40;
            }
            v42 = *(_DWORD *)(a1 + 248);
            if ( v41 )
              v24 = v41;
            ++*(_QWORD *)(a1 + 232);
            v36 = v42 + 1;
            if ( 4 * v36 < 3 * v31 )
            {
              if ( v31 - *(_DWORD *)(a1 + 252) - v36 <= v31 >> 3 )
              {
                sub_13C4D80(v55, v31);
                v43 = *(_DWORD *)(a1 + 256);
                if ( !v43 )
                {
LABEL_84:
                  ++*(_DWORD *)(a1 + 248);
                  BUG();
                }
                v44 = *((_QWORD *)v20 - 1);
                v45 = v43 - 1;
                v46 = 1;
                v47 = 0;
                v48 = *(_QWORD *)(a1 + 240);
                v36 = *(_DWORD *)(a1 + 248) + 1;
                v49 = (v43 - 1) & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
                v24 = (_QWORD *)(v48 + 16LL * v49);
                v50 = *v24;
                if ( v44 != *v24 )
                {
                  while ( v50 != -8 )
                  {
                    if ( v50 == -16 && !v47 )
                      v47 = v24;
                    v49 = v45 & (v46 + v49);
                    v24 = (_QWORD *)(v48 + 16LL * v49);
                    v50 = *v24;
                    if ( v44 == *v24 )
                      goto LABEL_39;
                    ++v46;
                  }
LABEL_54:
                  if ( v47 )
                    v24 = v47;
                }
              }
LABEL_39:
              *(_DWORD *)(a1 + 248) = v36;
              if ( *v24 != -8 )
                --*(_DWORD *)(a1 + 252);
              v39 = *((_QWORD *)v20 - 1);
              v24[1] = 0;
              *v24 = v39;
              goto LABEL_31;
            }
LABEL_37:
            sub_13C4D80(v55, 2 * v31);
            v32 = *(_DWORD *)(a1 + 256);
            if ( !v32 )
              goto LABEL_84;
            v33 = *((_QWORD *)v20 - 1);
            v34 = v32 - 1;
            v35 = *(_QWORD *)(a1 + 240);
            v36 = *(_DWORD *)(a1 + 248) + 1;
            v37 = (v32 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
            v24 = (_QWORD *)(v35 + 16LL * v37);
            v38 = *v24;
            if ( v33 != *v24 )
            {
              v54 = 1;
              v47 = 0;
              while ( v38 != -8 )
              {
                if ( v38 == -16 && !v47 )
                  v47 = v24;
                v37 = v34 & (v54 + v37);
                v24 = (_QWORD *)(v35 + 16LL * v37);
                v38 = *v24;
                if ( v33 == *v24 )
                  goto LABEL_39;
                ++v54;
              }
              goto LABEL_54;
            }
            goto LABEL_39;
          }
LABEL_31:
          v24[1] = a2;
          v26 = v58;
          v27 = *(_QWORD *)(a1 + 328);
          v28 = (_QWORD *)sub_22077B0(64);
          v29 = *((_QWORD *)v26 - 1);
          v28[3] = 2;
          v28[4] = 0;
          v28[5] = v29;
          if ( v29 != -8 && v29 != 0 && v29 != -16 )
            sub_164C220(v28 + 3);
          v28[6] = a1;
          v28[7] = 0;
          v28[2] = &unk_49EA488;
          sub_2208C80(v28, v27);
          v30 = *(_QWORD *)(a1 + 328);
          ++*(_QWORD *)(a1 + 344);
          *(_QWORD *)(v30 + 56) = v30;
          v20 = v58 - 8;
          v58 = v20;
          if ( v57 == v20 )
            goto LABEL_12;
        }
        ++*(_QWORD *)(a1 + 232);
        goto LABEL_37;
      }
      goto LABEL_12;
    }
LABEL_7:
    v8 = 0;
    goto LABEL_8;
  }
  v5 = a2[1];
  if ( v5 )
  {
LABEL_4:
    while ( 1 )
    {
      v6 = sub_1648700(v5);
      v7 = *(_BYTE *)(v6 + 16);
      if ( v7 <= 0x17u )
        goto LABEL_7;
      if ( v7 == 54 )
      {
        if ( (unsigned __int8)sub_13C13B0(a1, (_QWORD *)v6, 0, 0, 0) )
          goto LABEL_7;
      }
      else
      {
        if ( v7 != 55 )
          goto LABEL_7;
        v15 = *(_QWORD *)(v6 - 48);
        if ( !v15 )
          BUG();
        if ( (_QWORD *)v15 == a2 )
          goto LABEL_7;
        if ( *(_BYTE *)(v15 + 16) != 15 )
        {
          v16 = sub_1632FA0(a2[5]);
          v17 = (_QWORD *)sub_14AD280(*(_QWORD *)(v6 - 48), v16, 6);
          v18 = *(_QWORD *)(a1 + 16);
          v56 = v17;
          if ( !(unsigned __int8)sub_140B1C0(v17, v18, 0) || (unsigned __int8)sub_13C13B0(a1, v56, 0, 0, a2) )
            goto LABEL_7;
          v19 = v58;
          if ( v58 == v59 )
          {
            sub_1287830((__int64)&v57, v58, &v56);
          }
          else
          {
            if ( v58 )
            {
              *(_QWORD *)v58 = v56;
              v19 = v58;
            }
            v58 = v19 + 8;
          }
        }
      }
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_28;
    }
  }
LABEL_12:
  v10 = *(_QWORD **)(a1 + 136);
  if ( *(_QWORD **)(a1 + 144) != v10 )
    goto LABEL_13;
  v51 = &v10[*(unsigned int *)(a1 + 156)];
  v52 = *(_DWORD *)(a1 + 156);
  if ( v10 != v51 )
  {
    v53 = 0;
    while ( (_QWORD *)*v10 != a2 )
    {
      if ( *v10 == -2 )
        v53 = v10;
      if ( v51 == ++v10 )
      {
        if ( !v53 )
          goto LABEL_65;
        *v53 = a2;
        --*(_DWORD *)(a1 + 160);
        ++*(_QWORD *)(a1 + 128);
        goto LABEL_14;
      }
    }
    goto LABEL_14;
  }
LABEL_65:
  if ( v52 < *(_DWORD *)(a1 + 152) )
  {
    *(_DWORD *)(a1 + 156) = v52 + 1;
    *v51 = a2;
    ++*(_QWORD *)(a1 + 128);
  }
  else
  {
LABEL_13:
    sub_16CCBA0(a1 + 128, a2);
  }
LABEL_14:
  v11 = *(_QWORD *)(a1 + 328);
  v12 = (_QWORD *)sub_22077B0(64);
  v12[3] = 2;
  v13 = v12;
  v12[4] = 0;
  v12[5] = a2;
  if ( a2 != (_QWORD *)-16LL && a2 != (_QWORD *)-8LL )
    sub_164C220(v12 + 3);
  v13[6] = a1;
  v13[2] = &unk_49EA488;
  v13[7] = 0;
  sub_2208C80(v13, v11);
  v14 = *(_QWORD *)(a1 + 328);
  ++*(_QWORD *)(a1 + 344);
  v8 = 1;
  *(_QWORD *)(v14 + 56) = v14;
LABEL_8:
  if ( v57 )
    j_j___libc_free_0(v57, v59 - v57);
  return v8;
}

// Function: sub_3936BD0
// Address: 0x3936bd0
//
void __fastcall sub_3936BD0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  int v5; // ebx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // r13d
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rdx
  unsigned int v25; // eax
  __int64 *v26; // rsi
  __int64 v27; // rsi
  __int64 v28; // rdx
  unsigned int v29; // eax
  __int64 *v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 *v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rdx
  unsigned int v37; // eax
  __int64 *v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  unsigned int v41; // eax
  __int64 *v42; // rsi
  __int64 v43; // rsi
  __int64 v44; // rcx
  __int64 v45; // rax
  __int64 v46; // rdx
  _QWORD *v47; // rax
  __int64 v48; // rcx
  _QWORD *v49; // rdx
  __int64 v50; // rdi
  __int64 v51; // rsi
  int v52; // edx
  char v53; // cl
  __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rax
  _QWORD *v58; // rsi
  _QWORD *v59; // rcx
  __int64 v60; // rax
  _QWORD *v61; // rdx
  int v62; // eax
  int v63; // edx
  int v64; // edi
  __int64 v65; // [rsp+0h] [rbp-60h]
  __int64 v66; // [rsp+8h] [rbp-58h]
  int v67; // [rsp+14h] [rbp-4Ch]
  __int64 v68; // [rsp+18h] [rbp-48h]
  __int64 v69; // [rsp+20h] [rbp-40h]
  char v70; // [rsp+2Ah] [rbp-36h]
  char v71; // [rsp+2Bh] [rbp-35h]
  int v72; // [rsp+2Ch] [rbp-34h]

  v2 = *(unsigned int *)(a1 + 8);
  v72 = v2;
  if ( (int)v2 <= 0 )
    return;
  v68 = 0;
  v5 = 0;
  v69 = 0;
  v66 = 0;
  v65 = 0;
  v67 = 0;
  v70 = 0;
  v71 = 0;
  while ( 1 )
  {
    v20 = v5 + 1;
    v21 = *(_QWORD *)(a1 + 8 * (v5 - v2));
    v22 = sub_161E970(v21);
    if ( v23 == 10 && *(_QWORD *)v22 == 0x7261507473726966LL && *(_WORD *)(v22 + 8) == 28001 )
      break;
    v6 = sub_161E970(v21);
    if ( v7 == 9 && *(_QWORD *)v6 == 0x6D617261506D756ELL && *(_BYTE *)(v6 + 8) == 115 )
    {
      v28 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
      v29 = *(_DWORD *)(v28 + 32);
      v30 = *(__int64 **)(v28 + 24);
      if ( v29 <= 0x40 )
        v31 = (__int64)((_QWORD)v30 << (64 - (unsigned __int8)v29)) >> (64 - (unsigned __int8)v29);
      else
        v31 = *v30;
      sub_39367D0(a2, v31);
    }
    else
    {
      v8 = sub_161E970(v21);
      if ( v9 == 12 && *(_QWORD *)v8 == 0x7465527473726966LL && *(_DWORD *)(v8 + 8) == 1433301621 )
      {
        v32 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
        v33 = *(_DWORD *)(v32 + 32);
        v34 = *(__int64 **)(v32 + 24);
        if ( v33 <= 0x40 )
          v35 = (__int64)((_QWORD)v34 << (64 - (unsigned __int8)v33)) >> (64 - (unsigned __int8)v33);
        else
          v35 = *v34;
        sub_3936800(a2, v35);
      }
      else
      {
        v10 = sub_161E970(v21);
        if ( v11 == 11
          && *(_QWORD *)v10 == 0x7465527473726966LL
          && *(_WORD *)(v10 + 8) == 29301
          && *(_BYTE *)(v10 + 10) == 110 )
        {
          v36 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
          v37 = *(_DWORD *)(v36 + 32);
          v38 = *(__int64 **)(v36 + 24);
          if ( v37 <= 0x40 )
            v39 = (__int64)((_QWORD)v38 << (64 - (unsigned __int8)v37)) >> (64 - (unsigned __int8)v37);
          else
            v39 = *v38;
          sub_39367F0(a2, v39);
        }
        else
        {
          v12 = sub_161E970(v21);
          if ( v13 == 11
            && *(_QWORD *)v12 == 0x78614D6C61636F6CLL
            && *(_WORD *)(v12 + 8) == 25938
            && *(_BYTE *)(v12 + 10) == 103 )
          {
            v40 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
            v41 = *(_DWORD *)(v40 + 32);
            v42 = *(__int64 **)(v40 + 24);
            if ( v41 <= 0x40 )
              v43 = (__int64)((_QWORD)v42 << (64 - (unsigned __int8)v41)) >> (64 - (unsigned __int8)v41);
            else
              v43 = *v42;
            sub_3936840(a2, v43);
          }
          else
          {
            v14 = (_QWORD *)sub_161E970(v21);
            if ( v15 == 8 && *v14 == 0x5268637461726373LL )
            {
              v44 = *(unsigned int *)(a1 + 8);
              v45 = *(_QWORD *)(a1 + 8 * (v20 - v44));
              if ( !v45 || (v46 = *(_QWORD *)(v45 + 136)) == 0 )
              {
                v68 = 0;
                v69 = 0;
                v66 = 0;
                v65 = 0;
                v71 = 1;
                goto LABEL_10;
              }
              v47 = *(_QWORD **)(v46 + 24);
              if ( *(_DWORD *)(v46 + 32) > 0x40u )
                v47 = (_QWORD *)*v47;
              v20 = v5 + 2;
              v48 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v5 + 2 - v44)) + 136LL);
              v49 = *(_QWORD **)(v48 + 24);
              if ( *(_DWORD *)(v48 + 32) > 0x40u )
                v49 = (_QWORD *)*v49;
              if ( (int)v47 > (int)v49 )
              {
                v71 = 1;
                goto LABEL_10;
              }
              v50 = v69;
              v51 = v68;
              v52 = (_DWORD)v49 + 1;
              while ( (int)v47 > 63 )
              {
                if ( (int)v47 > 127 )
                {
                  if ( (int)v47 > 191 )
                  {
                    if ( (int)v47 <= 255 )
                      v65 |= 1LL << ((unsigned __int8)v47 + 64);
                  }
                  else
                  {
                    v66 |= 1LL << ((unsigned __int8)v47 + 0x80);
                  }
LABEL_54:
                  LODWORD(v47) = (_DWORD)v47 + 1;
                  if ( v52 == (_DWORD)v47 )
                    goto LABEL_58;
                  continue;
                }
                v53 = (_BYTE)v47 - 64;
                LODWORD(v47) = (_DWORD)v47 + 1;
                v50 |= 1LL << v53;
                if ( v52 == (_DWORD)v47 )
                {
LABEL_58:
                  v69 = v50;
                  v68 = v51;
                  v71 = 1;
                  goto LABEL_10;
                }
              }
              v51 |= 1LL << (char)v47;
              goto LABEL_54;
            }
            v16 = sub_161E970(v21);
            if ( v17 == 9 && *(_QWORD *)v16 == 0x4368637461726373LL && *(_BYTE *)(v16 + 8) == 66 )
            {
              v54 = *(unsigned int *)(a1 + 8);
              v55 = *(_QWORD *)(a1 + 8 * (v20 - v54));
              if ( v55 && (v56 = *(_QWORD *)(v55 + 136)) != 0 )
              {
                v59 = *(_QWORD **)(v56 + 24);
                if ( *(_DWORD *)(v56 + 32) > 0x40u )
                  v59 = (_QWORD *)*v59;
                v20 = v5 + 2;
                v60 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v5 + 2 - v54)) + 136LL);
                v61 = *(_QWORD **)(v60 + 24);
                if ( *(_DWORD *)(v60 + 32) > 0x40u )
                  v61 = (_QWORD *)*v61;
                if ( (int)v59 <= (int)v61 )
                {
                  v62 = v67;
                  v63 = (_DWORD)v61 + 1;
                  do
                  {
                    v64 = 1 << (char)v59;
                    LODWORD(v59) = (_DWORD)v59 + 1;
                    v62 |= v64;
                  }
                  while ( v63 != (_DWORD)v59 );
                  v67 = v62;
                }
                v70 = 1;
              }
              else
              {
                v67 = 0;
                v70 = 1;
              }
            }
            else
            {
              v18 = sub_161E970(v21);
              if ( v19 == 10 && *(_QWORD *)v18 == 0x69747265706F7270LL && *(_WORD *)(v18 + 8) == 29541 )
              {
                v57 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
                v58 = *(_QWORD **)(v57 + 24);
                if ( *(_DWORD *)(v57 + 32) > 0x40u )
                  v58 = (_QWORD *)*v58;
                sub_3936850(a2, (int)v58);
              }
            }
          }
        }
      }
    }
LABEL_10:
    v5 = v20 + 1;
    if ( v72 <= v20 + 1 )
      goto LABEL_18;
LABEL_11:
    v2 = *(unsigned int *)(a1 + 8);
  }
  v24 = *(_QWORD *)(*(_QWORD *)(a1 + 8 * (v20 - (unsigned __int64)*(unsigned int *)(a1 + 8))) + 136LL);
  v25 = *(_DWORD *)(v24 + 32);
  v26 = *(__int64 **)(v24 + 24);
  if ( v25 <= 0x40 )
    v27 = (__int64)((_QWORD)v26 << (64 - (unsigned __int8)v25)) >> (64 - (unsigned __int8)v25);
  else
    v27 = *v26;
  v5 += 2;
  sub_39367E0(a2, v27);
  if ( v72 > v5 )
    goto LABEL_11;
LABEL_18:
  if ( v70 )
    sub_3936810(a2, v67);
  if ( v71 )
    sub_3936820(a2, v65, v66, v69, v68);
}

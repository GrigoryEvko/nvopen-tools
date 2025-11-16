// Function: sub_1BA7B00
// Address: 0x1ba7b00
//
__int64 __fastcall sub_1BA7B00(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  _QWORD *v5; // r8
  unsigned int v7; // eax
  __int64 v8; // r9
  int v9; // ecx
  int v10; // r10d
  unsigned int v11; // edx
  __int64 **v12; // rdi
  __int64 v13; // rdx
  __int64 **v14; // rsi
  int v15; // edx
  char v16; // al
  __int64 **v17; // rbx
  __int64 v18; // rax
  __int64 **v19; // r8
  __int64 *v20; // r14
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned int v24; // esi
  __int64 **v25; // rcx
  __int64 v26; // r8
  unsigned int v27; // edx
  __int64 v28; // rax
  __int64 **v29; // rdi
  __int64 v31; // rdi
  int v32; // r8d
  int v33; // r9d
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 *v36; // r14
  __int64 v37; // r13
  __int64 v38; // rbx
  __int64 v39; // rax
  _QWORD *v40; // rax
  __int64 v41; // rsi
  _QWORD *v42; // rdx
  __int64 v43; // rdi
  __int64 *v44; // r12
  __int64 *v45; // rax
  __int64 *v46; // r12
  __int64 *v47; // rax
  int v48; // ebx
  unsigned int v49; // eax
  __int64 v50; // rax
  int v51; // r12d
  __int64 v52; // r11
  int v53; // edi
  __int64 v54; // r14
  __int64 **v55; // [rsp+8h] [rbp-108h]
  __int64 *v56; // [rsp+10h] [rbp-100h]
  __int64 v57; // [rsp+18h] [rbp-F8h]
  __int64 v58; // [rsp+20h] [rbp-F0h]
  __int64 v59; // [rsp+28h] [rbp-E8h]
  int v60; // [rsp+30h] [rbp-E0h]
  _QWORD *v61; // [rsp+38h] [rbp-D8h]
  unsigned int v62; // [rsp+44h] [rbp-CCh]
  __int64 v64; // [rsp+50h] [rbp-C0h]
  unsigned int v66; // [rsp+68h] [rbp-A8h]
  unsigned int v67; // [rsp+6Ch] [rbp-A4h]
  unsigned int v68; // [rsp+7Ch] [rbp-94h] BYREF
  __int64 **v69; // [rsp+80h] [rbp-90h] BYREF
  __int64 v70; // [rsp+88h] [rbp-88h] BYREF
  _QWORD *v71; // [rsp+90h] [rbp-80h] BYREF
  __int64 v72; // [rsp+98h] [rbp-78h]
  _QWORD v73[14]; // [rsp+A0h] [rbp-70h] BYREF

  v5 = v73;
  v72 = 0x800000001LL;
  v7 = 1;
  v71 = v73;
  v73[0] = a2;
  v62 = 0;
  v59 = a1 + 168;
  do
  {
    while ( 1 )
    {
      v13 = v7--;
      v14 = (__int64 **)v5[v13 - 1];
      LODWORD(v72) = v7;
      v15 = *(_DWORD *)(a3 + 24);
      v69 = v14;
      if ( !v15 )
        goto LABEL_5;
      v8 = *(_QWORD *)(a3 + 8);
      v9 = v15 - 1;
      v10 = 1;
      v11 = (v15 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v12 = *(__int64 ***)(v8 + 16LL * v11);
      if ( v14 != v12 )
        break;
LABEL_3:
      if ( !v7 )
        goto LABEL_22;
    }
    while ( v12 != (__int64 **)-8LL )
    {
      v11 = v9 & (v10 + v11);
      v12 = *(__int64 ***)(v8 + 16LL * v11);
      if ( v14 == v12 )
        goto LABEL_3;
      ++v10;
    }
LABEL_5:
    v60 = sub_1BA7710(a1, (__int64)v14, a4);
    v66 = a4 * sub_1BA7710(a1, (__int64)v69, 1);
    v16 = sub_1B91FD0(a1, (__int64)v69);
    v17 = v69;
    if ( v16 && *((_BYTE *)*v69 + 8) )
    {
      v46 = *(__int64 **)(a1 + 328);
      v47 = sub_1B8E090(*v69, a4);
      v48 = sub_14A2E40(v46, (__int64)v47, 1u, 0) + v66;
      v49 = v48 + a4 * sub_14A3410(*(_QWORD *)(a1 + 328));
      v17 = v69;
      v66 = v49;
    }
    v18 = 24LL * (*((_DWORD *)v17 + 5) & 0xFFFFFFF);
    if ( (*((_BYTE *)v17 + 23) & 0x40) != 0 )
    {
      v19 = (__int64 **)*(v17 - 1);
      v17 = &v19[(unsigned __int64)v18 / 8];
    }
    else
    {
      v19 = &v17[v18 / 0xFFFFFFFFFFFFFFF8LL];
    }
    if ( v19 != v17 )
    {
      v67 = a4;
      v20 = (__int64 *)v19;
      v58 = a1 + 200;
      while ( 1 )
      {
        v21 = *v20;
        if ( *(_BYTE *)(*v20 + 16) > 0x17u )
        {
          v22 = *(_QWORD *)(v21 + 8);
          v23 = *(_QWORD *)(v21 + 40);
          if ( v22 && !*(_QWORD *)(v22 + 8) && *(_QWORD *)(a2 + 40) == v23 )
          {
            v68 = v67;
            if ( v67 == 1 )
            {
              sub_1377F70(*(_QWORD *)(a1 + 296) + 56LL, v23);
            }
            else
            {
              if ( (unsigned __int8)sub_1B97860(v58, (int *)&v68, &v70) )
                v31 = v70;
              else
                v31 = *(_QWORD *)(a1 + 208) + 80LL * *(unsigned int *)(a1 + 224);
              if ( !sub_13A0E30(v31 + 8, v21) && !(unsigned __int8)sub_1B91FD0(a1, v21) )
              {
                v34 = 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF);
                if ( (*(_BYTE *)(v21 + 23) & 0x40) != 0 )
                {
                  v35 = *(_QWORD *)(v21 - 8);
                  v64 = v35 + v34;
                }
                else
                {
                  v64 = v21;
                  v35 = v21 - v34;
                }
                if ( v35 == v64 )
                {
LABEL_63:
                  v50 = (unsigned int)v72;
                  if ( (unsigned int)v72 >= HIDWORD(v72) )
                  {
                    sub_16CD150((__int64)&v71, v73, 0, 8, v32, v33);
                    v50 = (unsigned int)v72;
                  }
                  v71[v50] = v21;
                  LODWORD(v72) = v72 + 1;
                  goto LABEL_17;
                }
                v57 = v21;
                v55 = v17;
                v56 = v20;
                v36 = (__int64 *)v35;
                while ( 2 )
                {
                  v37 = *v36;
                  if ( *(_BYTE *)(*v36 + 16) > 0x17u )
                  {
                    v68 = v67;
                    if ( (unsigned __int8)sub_1B97860(v59, (int *)&v68, &v70) )
                    {
                      v38 = v70;
                      v39 = *(_QWORD *)(v70 + 24);
                      if ( v39 != *(_QWORD *)(v70 + 16) )
                        goto LABEL_41;
LABEL_67:
                      v61 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(v38 + 36));
                    }
                    else
                    {
                      v38 = 80LL * *(unsigned int *)(a1 + 192) + *(_QWORD *)(a1 + 176);
                      v39 = *(_QWORD *)(v38 + 24);
                      if ( v39 == *(_QWORD *)(v38 + 16) )
                        goto LABEL_67;
LABEL_41:
                      v61 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(v38 + 32));
                    }
                    v40 = sub_15CC2D0(v38 + 8, v37);
                    v41 = *(_QWORD *)(v38 + 24);
                    if ( v41 == *(_QWORD *)(v38 + 16) )
                      v42 = (_QWORD *)(v41 + 8LL * *(unsigned int *)(v38 + 36));
                    else
                      v42 = (_QWORD *)(v41 + 8LL * *(unsigned int *)(v38 + 32));
                    for ( ; v42 != v40; ++v40 )
                    {
                      if ( *v40 < 0xFFFFFFFFFFFFFFFELL )
                        break;
                    }
                    if ( v40 != v61 )
                    {
                      v21 = v57;
                      v20 = v56;
                      v17 = v55;
                      break;
                    }
                  }
                  v36 += 3;
                  if ( (__int64 *)v64 == v36 )
                  {
                    v21 = v57;
                    v20 = v56;
                    v17 = v55;
                    goto LABEL_63;
                  }
                  continue;
                }
              }
              if ( sub_1377F70(*(_QWORD *)(a1 + 296) + 56LL, *(_QWORD *)(v21 + 40)) )
              {
                v68 = v67;
LABEL_52:
                if ( (unsigned __int8)sub_1B97860(v58, (int *)&v68, &v70) )
                  v43 = v70;
                else
                  v43 = *(_QWORD *)(a1 + 208) + 80LL * *(unsigned int *)(a1 + 224);
                if ( !sub_13A0E30(v43 + 8, v21) )
                {
                  v44 = *(__int64 **)(a1 + 328);
                  v45 = sub_1B8E090(*(__int64 **)v21, v67);
                  v66 += sub_14A2E40(v44, (__int64)v45, 0, 1u);
                }
              }
            }
          }
          else if ( sub_1377F70(*(_QWORD *)(a1 + 296) + 56LL, v23) )
          {
            v68 = v67;
            if ( v67 != 1 )
              goto LABEL_52;
          }
        }
LABEL_17:
        v20 += 3;
        if ( v17 == (__int64 **)v20 )
        {
          a4 = v67;
          break;
        }
      }
    }
    v62 = v60 + v62 - (v66 >> 1);
    v24 = *(_DWORD *)(a3 + 24);
    if ( !v24 )
    {
      ++*(_QWORD *)a3;
LABEL_79:
      v54 = a3;
      v24 *= 2;
LABEL_80:
      sub_14672C0(v54, v24);
      sub_1463AD0(v54, (__int64 *)&v69, &v70);
      v28 = v70;
      v25 = v69;
      v53 = *(_DWORD *)(v54 + 16) + 1;
      goto LABEL_75;
    }
    v25 = v69;
    v26 = *(_QWORD *)(a3 + 8);
    v27 = (v24 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
    v28 = v26 + 16LL * v27;
    v29 = *(__int64 ***)v28;
    if ( *(__int64 ***)v28 == v69 )
      goto LABEL_21;
    v51 = 1;
    v52 = 0;
    while ( v29 != (__int64 **)-8LL )
    {
      if ( v29 == (__int64 **)-16LL && !v52 )
        v52 = v28;
      v27 = (v24 - 1) & (v51 + v27);
      v28 = v26 + 16LL * v27;
      v29 = *(__int64 ***)v28;
      if ( v69 == *(__int64 ***)v28 )
        goto LABEL_21;
      ++v51;
    }
    if ( v52 )
      v28 = v52;
    ++*(_QWORD *)a3;
    v53 = *(_DWORD *)(a3 + 16) + 1;
    if ( 4 * v53 >= 3 * v24 )
      goto LABEL_79;
    if ( v24 - *(_DWORD *)(a3 + 20) - v53 <= v24 >> 3 )
    {
      v54 = a3;
      goto LABEL_80;
    }
LABEL_75:
    *(_DWORD *)(a3 + 16) = v53;
    if ( *(_QWORD *)v28 != -8 )
      --*(_DWORD *)(a3 + 20);
    *(_QWORD *)v28 = v25;
    *(_DWORD *)(v28 + 8) = 0;
LABEL_21:
    *(_DWORD *)(v28 + 8) = v66 >> 1;
    v7 = v72;
    v5 = v71;
  }
  while ( (_DWORD)v72 );
LABEL_22:
  if ( v5 != v73 )
    _libc_free((unsigned __int64)v5);
  return v62;
}

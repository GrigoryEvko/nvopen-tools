// Function: sub_24005F0
// Address: 0x24005f0
//
__int64 __fastcall sub_24005F0(unsigned __int8 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r15d
  __int64 v7; // rax
  __int64 v11; // r8
  unsigned int v12; // esi
  __int64 v13; // rdx
  unsigned __int8 *v14; // r11
  int v16; // eax
  __int64 v17; // rsi
  int v18; // edx
  unsigned int v19; // eax
  unsigned __int8 *v20; // r8
  int v21; // edx
  int v22; // r14d
  unsigned int v23; // eax
  __int64 v24; // rcx
  __int64 v25; // r9
  unsigned __int8 v26; // al
  __int64 v27; // r9
  __int64 v28; // rcx
  unsigned __int8 *v29; // r11
  __int64 *v30; // r8
  __int64 v31; // r14
  __int64 v32; // r15
  unsigned __int8 *v33; // r12
  __int64 *v34; // rax
  int v35; // r10d
  __int64 *v36; // r15
  __int64 *v37; // r14
  __int64 v38; // r13
  __int64 v39; // r12
  __int64 *v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  __int64 *v43; // rdx
  int v44; // eax
  __int64 v45; // [rsp+0h] [rbp-A0h]
  unsigned __int8 v46; // [rsp+Eh] [rbp-92h]
  unsigned __int8 v47; // [rsp+Fh] [rbp-91h]
  __int64 v49; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v50; // [rsp+10h] [rbp-90h]
  __int64 v52; // [rsp+18h] [rbp-88h]
  __int64 *v53; // [rsp+18h] [rbp-88h]
  __int64 v54; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v55; // [rsp+28h] [rbp-78h] BYREF
  __int64 *v56; // [rsp+30h] [rbp-70h] BYREF
  __int64 *v57; // [rsp+38h] [rbp-68h] BYREF
  __int64 v58; // [rsp+40h] [rbp-60h] BYREF
  __int64 *v59; // [rsp+48h] [rbp-58h]
  __int64 v60; // [rsp+50h] [rbp-50h]
  __int64 v61; // [rsp+58h] [rbp-48h]

  v6 = 1;
  if ( *a1 > 0x1Cu )
  {
    v7 = *(unsigned int *)(a6 + 24);
    v55 = a1;
    if ( (_DWORD)v7 )
    {
      v11 = *(_QWORD *)(a6 + 8);
      v12 = (v7 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v13 = v11 + 16LL * v12;
      v14 = *(unsigned __int8 **)v13;
      if ( a1 == *(unsigned __int8 **)v13 )
      {
LABEL_4:
        if ( v13 != v11 + 16 * v7 )
          return *(unsigned __int8 *)(v13 + 8);
      }
      else
      {
        v21 = 1;
        while ( v14 != (unsigned __int8 *)-4096LL )
        {
          v22 = v21 + 1;
          v12 = (v7 - 1) & (v21 + v12);
          v13 = v11 + 16LL * v12;
          v14 = *(unsigned __int8 **)v13;
          if ( a1 == *(unsigned __int8 **)v13 )
            goto LABEL_4;
          v21 = v22;
        }
      }
    }
    v16 = *(_DWORD *)(a4 + 24);
    v17 = *(_QWORD *)(a4 + 8);
    if ( v16 )
    {
      v18 = v16 - 1;
      v19 = (v16 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v20 = *(unsigned __int8 **)(v17 + 8LL * v19);
      if ( a1 == v20 )
      {
LABEL_9:
        v6 = 0;
        *(_BYTE *)sub_23FEC80(a6, (__int64 *)&v55) = 0;
        return v6;
      }
      v35 = 1;
      while ( v20 != (unsigned __int8 *)-4096LL )
      {
        v19 = v18 & (v35 + v19);
        v20 = *(unsigned __int8 **)(v17 + 8LL * v19);
        if ( a1 == v20 )
          goto LABEL_9;
        ++v35;
      }
    }
    v23 = sub_B19DB0(a3, (__int64)a1, a2);
    v24 = a4;
    v25 = a6;
    v6 = v23;
    if ( !(_BYTE)v23 )
    {
      v52 = a6;
      v49 = v24;
      v26 = sub_23FAAE0(v55, a3);
      v27 = v52;
      v47 = v26;
      if ( v26 )
      {
        v58 = 0;
        v59 = 0;
        v28 = v49;
        v60 = 0;
        v61 = 0;
        if ( (v55[7] & 0x40) != 0 )
        {
          v29 = (unsigned __int8 *)*((_QWORD *)v55 - 1);
          v50 = &v29[32 * (*((_DWORD *)v55 + 1) & 0x7FFFFFF)];
        }
        else
        {
          v50 = v55;
          v29 = &v55[-32 * (*((_DWORD *)v55 + 1) & 0x7FFFFFF)];
        }
        if ( v50 == v29 )
          goto LABEL_26;
        v46 = v6;
        v30 = &v58;
        v31 = v28;
        v32 = v52;
        v45 = a5;
        v33 = v29;
        while ( 1 )
        {
          v53 = v30;
          if ( !(unsigned __int8)sub_24005F0(*(_QWORD *)v33, a2, a3, v31, v30, v32) )
            break;
          v33 += 32;
          v30 = v53;
          if ( v50 == v33 )
          {
            a5 = v45;
            v27 = v32;
LABEL_26:
            if ( !a5 )
              goto LABEL_28;
            v34 = v59;
            if ( !(_DWORD)v60 )
              goto LABEL_28;
            v36 = &v59[(unsigned int)v61];
            if ( v59 == v36 )
              goto LABEL_28;
            while ( 1 )
            {
              v37 = v34;
              if ( *v34 != -8192 && *v34 != -4096 )
                break;
              if ( v36 == ++v34 )
                goto LABEL_28;
            }
            if ( v36 == v34 )
            {
LABEL_28:
              *(_BYTE *)sub_23FEC80(v27, (__int64 *)&v55) = 1;
              sub_C7D6A0((__int64)v59, 8LL * (unsigned int)v61, 8);
              return v47;
            }
            v38 = a5;
            v39 = v27;
            while ( 2 )
            {
              if ( (unsigned __int8)sub_23FDF60(v38, v37, &v56) )
              {
LABEL_44:
                v40 = v37 + 1;
                if ( v36 == v37 + 1 )
                  goto LABEL_48;
                while ( 1 )
                {
                  v37 = v40;
                  if ( *v40 != -8192 && *v40 != -4096 )
                    break;
                  if ( v36 == ++v40 )
                    goto LABEL_48;
                }
                if ( v36 == v40 )
                {
LABEL_48:
                  v27 = v39;
                  goto LABEL_28;
                }
                continue;
              }
              break;
            }
            v41 = *(_DWORD *)(v38 + 24);
            v42 = *(_DWORD *)(v38 + 16);
            v43 = v56;
            ++*(_QWORD *)v38;
            v44 = v42 + 1;
            v57 = v43;
            if ( 4 * v44 >= 3 * v41 )
            {
              v41 *= 2;
            }
            else if ( v41 - *(_DWORD *)(v38 + 20) - v44 > v41 >> 3 )
            {
LABEL_53:
              *(_DWORD *)(v38 + 16) = v44;
              if ( *v43 != -4096 )
                --*(_DWORD *)(v38 + 20);
              *v43 = *v37;
              goto LABEL_44;
            }
            sub_CF4090(v38, v41);
            sub_23FDF60(v38, v37, &v57);
            v43 = v57;
            v44 = *(_DWORD *)(v38 + 16) + 1;
            goto LABEL_53;
          }
        }
        v54 = v32;
        v6 = v46;
        sub_C7D6A0((__int64)v59, 8LL * (unsigned int)v61, 8);
        v27 = v54;
      }
      *(_BYTE *)sub_23FEC80(v27, (__int64 *)&v55) = 0;
      return v6;
    }
    if ( a5 )
    {
      sub_2400480((__int64)&v58, a5, (__int64 *)&v55);
      v25 = a6;
    }
    *(_BYTE *)sub_23FEC80(v25, (__int64 *)&v55) = 1;
  }
  return v6;
}

// Function: sub_22CC150
// Address: 0x22cc150
//
__int64 __fastcall sub_22CC150(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int8 v8; // bl
  unsigned int v9; // esi
  unsigned __int8 v10; // bl
  unsigned int v11; // eax
  unsigned int v12; // esi
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 *v15; // rax
  __int64 v16; // r11
  char v17; // r15
  bool v18; // dl
  char v19; // bl
  char v20; // [rsp+0h] [rbp-1B0h]
  char v21; // [rsp+0h] [rbp-1B0h]
  unsigned __int8 *v22; // [rsp+10h] [rbp-1A0h]
  unsigned int v23; // [rsp+10h] [rbp-1A0h]
  char v24; // [rsp+10h] [rbp-1A0h]
  __int64 v25; // [rsp+20h] [rbp-190h] BYREF
  __int64 v26[3]; // [rsp+28h] [rbp-188h] BYREF
  __int64 v27; // [rsp+40h] [rbp-170h] BYREF
  unsigned int v28; // [rsp+48h] [rbp-168h]
  __int64 v29; // [rsp+50h] [rbp-160h] BYREF
  unsigned int v30; // [rsp+58h] [rbp-158h]
  __int64 v31; // [rsp+60h] [rbp-150h] BYREF
  unsigned int v32; // [rsp+68h] [rbp-148h]
  __int64 v33; // [rsp+70h] [rbp-140h] BYREF
  unsigned int v34; // [rsp+78h] [rbp-138h]
  __int64 v35[2]; // [rsp+80h] [rbp-130h] BYREF
  __int64 v36[2]; // [rsp+90h] [rbp-120h] BYREF
  __int64 v37[2]; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v38[2]; // [rsp+B0h] [rbp-100h] BYREF
  __int64 v39[2]; // [rsp+C0h] [rbp-F0h] BYREF
  __int64 v40[4]; // [rsp+D0h] [rbp-E0h] BYREF
  unsigned __int8 v41[8]; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v42; // [rsp+F8h] [rbp-B8h] BYREF
  unsigned int v43; // [rsp+100h] [rbp-B0h]
  __int64 v44; // [rsp+108h] [rbp-A8h] BYREF
  unsigned int v45; // [rsp+110h] [rbp-A0h]
  char v46; // [rsp+118h] [rbp-98h]
  unsigned __int8 v47[8]; // [rsp+120h] [rbp-90h] BYREF
  __int64 v48; // [rsp+128h] [rbp-88h] BYREF
  unsigned int v49; // [rsp+130h] [rbp-80h]
  __int64 v50; // [rsp+138h] [rbp-78h] BYREF
  unsigned int v51; // [rsp+140h] [rbp-70h]
  char v52; // [rsp+148h] [rbp-68h]
  __int64 v53; // [rsp+150h] [rbp-60h] BYREF
  unsigned int v54; // [rsp+158h] [rbp-58h]
  char v55; // [rsp+178h] [rbp-38h]

  sub_22C7100((__int64)v41, a2, *(_QWORD *)(a3 - 64), a4, a3);
  if ( !v46 )
  {
    *(_BYTE *)(a1 + 40) = 0;
    return a1;
  }
  sub_22C7100((__int64)v47, a2, *(_QWORD *)(a3 - 32), a4, a3);
  if ( v52 )
  {
    v8 = v41[0];
    if ( v41[0] != 4 )
    {
      if ( v41[0] != 5 )
      {
        if ( (unsigned __int8)(v47[0] - 4) > 1u )
          goto LABEL_26;
        v9 = sub_BCB060(*(_QWORD *)(a3 + 8));
        goto LABEL_12;
      }
      v9 = sub_BCB060(*(_QWORD *)(a3 + 8));
      if ( !sub_9876C0(&v42) )
      {
        v8 = v41[0];
LABEL_12:
        if ( v8 == 2 )
        {
          sub_AD8380((__int64)&v27, v42);
        }
        else if ( v8 )
        {
          sub_AADB10((__int64)&v27, v9, 1);
        }
        else
        {
          sub_AADB10((__int64)&v27, v9, 0);
        }
LABEL_15:
        v10 = v47[0];
        if ( v47[0] == 4
          || (v11 = sub_BCB060(*(_QWORD *)(a3 + 8)), v12 = v11, v10 == 5)
          && (v23 = v11, v15 = sub_9876C0(&v48), v10 = v47[0], v12 = v23, v15) )
        {
          v32 = v49;
          if ( v49 > 0x40 )
            sub_C43780((__int64)&v31, (const void **)&v48);
          else
            v31 = v48;
          v34 = v51;
          if ( v51 > 0x40 )
            sub_C43780((__int64)&v33, (const void **)&v50);
          else
            v33 = v50;
        }
        else if ( v10 == 2 )
        {
          sub_AD8380((__int64)&v31, v48);
        }
        else if ( v10 )
        {
          sub_AADB10((__int64)&v31, v12, 1);
        }
        else
        {
          sub_AADB10((__int64)&v31, v12, 0);
        }
        v25 = 0;
        v26[0] = 0;
        v13 = sub_99AEC0((_BYTE *)a3, &v25, v26, 0, 0);
        switch ( v13 )
        {
          case 0u:
            goto LABEL_25;
          case 7u:
            if ( v25 == *(_QWORD *)(a3 - 64) )
            {
              v19 = v41[0] == 5;
              sub_ABBBB0((__int64)v39, (__int64)&v27, 0);
              sub_22C06B0((__int64)&v53, (__int64)v39, v19);
              sub_22C0650(a1, (unsigned __int8 *)&v53);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v53);
              sub_969240(v40);
              sub_969240(v39);
            }
            else
            {
              if ( v25 != *(_QWORD *)(a3 - 32) )
              {
LABEL_25:
                sub_969240(&v33);
                sub_969240(&v31);
                sub_969240(&v29);
                sub_969240(&v27);
LABEL_26:
                v22 = *(unsigned __int8 **)(a3 - 96);
                if ( sub_98EF80(v22, *(_QWORD *)(a2 + 240), 0, 0, 0) )
                {
                  sub_22C9ED0((__int64)&v53, a2, *(_QWORD *)(a3 - 64), (__int64)v22, 1u, 0, 0);
                  sub_22EACA0(v39, v41, &v53);
                  sub_22C0090(v41);
                  sub_22C0650((__int64)v41, (unsigned __int8 *)v39);
                  sub_22C0090((unsigned __int8 *)v39);
                  v16 = (__int64)v22;
                  if ( v55 )
                  {
                    v55 = 0;
                    sub_22C0090((unsigned __int8 *)&v53);
                    v16 = (__int64)v22;
                  }
                  sub_22C9ED0((__int64)&v53, a2, *(_QWORD *)(a3 - 32), v16, 0, 0, 0);
                  sub_22EACA0(v39, v47, &v53);
                  sub_22C0090(v47);
                  sub_22C0650((__int64)v47, (unsigned __int8 *)v39);
                  sub_22C0090((unsigned __int8 *)v39);
                  if ( v55 )
                  {
                    v55 = 0;
                    sub_22C0090((unsigned __int8 *)&v53);
                  }
                }
                sub_22C05A0((__int64)&v53, v41);
                sub_22C0C70((__int64)&v53, (__int64)v47, 0, 0, 1u);
                sub_22C0650(a1, (unsigned __int8 *)&v53);
                *(_BYTE *)(a1 + 40) = 1;
                sub_22C0090((unsigned __int8 *)&v53);
                goto LABEL_28;
              }
              v17 = v47[0] == 5;
              sub_ABBBB0((__int64)v39, (__int64)&v31, 0);
              sub_22C06B0((__int64)&v53, (__int64)v39, v17);
              sub_22C0650(a1, (unsigned __int8 *)&v53);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v53);
              sub_969240(v40);
              sub_969240(v39);
            }
            break;
          case 8u:
            v54 = v28;
            if ( v28 > 0x40 )
              sub_C43690((__int64)&v53, 0, 0);
            else
              v53 = 0;
            sub_AADBC0((__int64)v35, &v53);
            sub_969240(&v53);
            if ( v25 == *(_QWORD *)(a3 - 64) )
            {
              v21 = v47[0] == 5;
              sub_ABBBB0((__int64)v37, (__int64)&v27, 0);
              sub_AB51C0((__int64)v39, (__int64)v35, (__int64)v37);
              sub_22C06B0((__int64)&v53, (__int64)v39, v21);
              sub_22C0650(a1, (unsigned __int8 *)&v53);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v53);
              sub_969240(v40);
              sub_969240(v39);
              sub_969240(v38);
              sub_969240(v37);
            }
            else
            {
              if ( v25 != *(_QWORD *)(a3 - 32) )
              {
                sub_969240(v36);
                sub_969240(v35);
                goto LABEL_25;
              }
              v20 = v47[0] == 5;
              sub_ABBBB0((__int64)v37, (__int64)&v31, 0);
              sub_AB51C0((__int64)v39, (__int64)v35, (__int64)v37);
              sub_22C06B0((__int64)&v53, (__int64)v39, v20);
              sub_22C0650(a1, (unsigned __int8 *)&v53);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v53);
              sub_969240(v40);
              sub_969240(v39);
              sub_969240(v38);
              sub_969240(v37);
            }
            sub_969240(v36);
            sub_969240(v35);
            break;
          default:
            v14 = *(_QWORD *)(a3 - 64);
            if ( v25 == v14 && *(_QWORD *)(a3 - 32) == v26[0] || v14 == v26[0] && v25 == *(_QWORD *)(a3 - 32) )
            {
              if ( v13 == 3 )
              {
                sub_AB5F70((__int64)v37, (__int64)&v27, (__int64)&v31);
              }
              else if ( v13 > 3 )
              {
                if ( v13 != 4 )
                  BUG();
                sub_AB6230((__int64)v37, (__int64)&v27, (__int64)&v31);
              }
              else if ( v13 == 1 )
              {
                sub_AB64F0((__int64)v37, (__int64)&v27, (__int64)&v31);
              }
              else
              {
                sub_AB6790((__int64)v37, (__int64)&v27, (__int64)&v31);
              }
              v18 = 1;
              if ( v41[0] != 5 )
                v18 = v47[0] == 5;
              v24 = v18;
              sub_AAF450((__int64)v39, (__int64)v37);
              sub_22C06B0((__int64)&v53, (__int64)v39, v24);
              sub_22C0650(a1, (unsigned __int8 *)&v53);
              *(_BYTE *)(a1 + 40) = 1;
              sub_22C0090((unsigned __int8 *)&v53);
              sub_969240(v40);
              sub_969240(v39);
              sub_969240(v38);
              sub_969240(v37);
              break;
            }
            goto LABEL_25;
        }
        sub_969240(&v33);
        sub_969240(&v31);
        sub_969240(&v29);
        sub_969240(&v27);
LABEL_28:
        if ( v52 )
        {
          v52 = 0;
          sub_22C0090(v47);
        }
        goto LABEL_6;
      }
    }
    v28 = v43;
    if ( v43 > 0x40 )
      sub_C43780((__int64)&v27, (const void **)&v42);
    else
      v27 = v42;
    v30 = v45;
    if ( v45 > 0x40 )
      sub_C43780((__int64)&v29, (const void **)&v44);
    else
      v29 = v44;
    goto LABEL_15;
  }
  *(_BYTE *)(a1 + 40) = 0;
LABEL_6:
  if ( v46 )
  {
    v46 = 0;
    sub_22C0090(v41);
  }
  return a1;
}

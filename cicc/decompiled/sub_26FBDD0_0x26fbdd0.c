// Function: sub_26FBDD0
// Address: 0x26fbdd0
//
__int64 __fastcall sub_26FBDD0(unsigned __int8 ***a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int8 **v8; // rdi
  __int64 v9; // rsi
  __int64 *v10; // rax
  __int64 result; // rax
  _QWORD *v12; // r12
  _QWORD *v13; // rbx
  unsigned __int64 v14; // rsi
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // rax
  _QWORD *v20; // rdi
  __int64 v21; // rcx
  __int64 v22; // rdx
  unsigned __int8 **v23; // rbx
  __int64 v24; // r12
  unsigned __int64 v25; // rbx
  int v26; // eax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rbx
  __int64 v32; // rbx
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 **v39; // rbx
  __int64 v40; // rax
  _DWORD *v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r15
  _QWORD *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int8 *v48; // rax
  char *v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r9
  __int64 v52; // r8
  __int64 v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rax
  __int64 v58; // r15
  __int64 v59; // rdx
  __int64 v60; // rsi
  unsigned __int8 *v61; // rsi
  __int64 v62; // [rsp+10h] [rbp-140h]
  __int64 v63; // [rsp+18h] [rbp-138h]
  __int64 v65; // [rsp+28h] [rbp-128h]
  __int64 v66; // [rsp+48h] [rbp-108h]
  int v67; // [rsp+58h] [rbp-F8h]
  __int64 v68[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v69; // [rsp+80h] [rbp-D0h]
  unsigned int *v70[2]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE v71[32]; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+C0h] [rbp-90h]
  __int64 v73; // [rsp+C8h] [rbp-88h]
  __int16 v74; // [rsp+D0h] [rbp-80h]
  __int64 v75; // [rsp+D8h] [rbp-78h]
  void **v76; // [rsp+E0h] [rbp-70h]
  void **v77; // [rsp+E8h] [rbp-68h]
  __int64 v78; // [rsp+F0h] [rbp-60h]
  int v79; // [rsp+F8h] [rbp-58h]
  __int16 v80; // [rsp+FCh] [rbp-54h]
  char v81; // [rsp+FEh] [rbp-52h]
  __int64 v82; // [rsp+100h] [rbp-50h]
  __int64 v83; // [rsp+108h] [rbp-48h]
  void *v84; // [rsp+110h] [rbp-40h] BYREF
  void *v85; // [rsp+118h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a2;
  v66 = *(_QWORD *)(a2 + 8);
  if ( v66 != *(_QWORD *)a2 )
  {
    while ( 1 )
    {
      v8 = *a1;
      v9 = *(_QWORD *)(v7 + 8);
      if ( !*((_BYTE *)*a1 + 204) )
        goto LABEL_12;
      v10 = (__int64 *)v8[23];
      a4 = *((unsigned int *)v8 + 49);
      a3 = &v10[a4];
      if ( v10 != a3 )
      {
        while ( v9 != *v10 )
        {
          if ( a3 == ++v10 )
            goto LABEL_60;
        }
        goto LABEL_7;
      }
LABEL_60:
      if ( (unsigned int)a4 < *((_DWORD *)v8 + 48) )
      {
        *((_DWORD *)v8 + 49) = a4 + 1;
        *a3 = v9;
        ++v8[22];
LABEL_13:
        v12 = sub_C52410();
        v13 = v12 + 1;
        v14 = sub_C959E0();
        v15 = (_QWORD *)v12[2];
        if ( v15 )
        {
          v16 = v12 + 1;
          do
          {
            while ( 1 )
            {
              v17 = v15[2];
              v18 = v15[3];
              if ( v14 <= v15[4] )
                break;
              v15 = (_QWORD *)v15[3];
              if ( !v18 )
                goto LABEL_18;
            }
            v16 = v15;
            v15 = (_QWORD *)v15[2];
          }
          while ( v17 );
LABEL_18:
          if ( v13 != v16 && v14 >= v16[4] )
            v13 = v16;
        }
        if ( v13 != (_QWORD *)((char *)sub_C52410() + 8) )
        {
          v19 = v13[7];
          if ( v19 )
          {
            v14 = (unsigned int)dword_4FF8F88;
            v20 = v13 + 6;
            do
            {
              while ( 1 )
              {
                v21 = *(_QWORD *)(v19 + 16);
                v22 = *(_QWORD *)(v19 + 24);
                if ( *(_DWORD *)(v19 + 32) >= dword_4FF8F88 )
                  break;
                v19 = *(_QWORD *)(v19 + 24);
                if ( !v22 )
                  goto LABEL_27;
              }
              v20 = (_QWORD *)v19;
              v19 = *(_QWORD *)(v19 + 16);
            }
            while ( v21 );
LABEL_27:
            if ( v13 + 6 != v20 && dword_4FF8F88 >= *((_DWORD *)v20 + 8) && *((int *)v20 + 9) > 0 )
            {
              result = (unsigned int)qword_4FF9008;
              if ( dword_4FF8D08 >= (unsigned int)qword_4FF9008 )
                return result;
            }
          }
        }
        v23 = *a1;
        if ( *((_BYTE *)*a1 + 104) )
        {
          v48 = sub_BD3990(*a1[1], v14);
          v49 = (char *)sub_BD5D20((__int64)v48);
          sub_26F96D0(
            v7,
            "single-impl",
            11,
            v49,
            v50,
            v51,
            (__int64 (__fastcall *)(__int64, __int64))v23[14],
            (__int64)v23[15]);
        }
        v24 = *(_QWORD *)(v7 + 8);
        ++dword_4FF8D08;
        v75 = sub_BD5C60(v24);
        v76 = &v84;
        v77 = &v85;
        v70[0] = (unsigned int *)v71;
        v84 = &unk_49DA100;
        v70[1] = (unsigned int *)0x200000000LL;
        v80 = 512;
        v74 = 0;
        v85 = &unk_49DA0B0;
        v78 = 0;
        v79 = 0;
        v81 = 7;
        v82 = 0;
        v83 = 0;
        v72 = 0;
        v73 = 0;
        sub_D5F1F0((__int64)v70, v24);
        v69 = 257;
        v25 = sub_26FAB50(
                (__int64 *)v70,
                0x31u,
                (unsigned __int64)*a1[1],
                *(__int64 ***)(*(_QWORD *)(v24 - 32) + 8LL),
                (__int64)v68,
                0,
                v67,
                0);
        v26 = dword_4FF8DA8;
        if ( dword_4FF8DA8 == 1 )
        {
          v69 = 257;
          v62 = sub_92B530(v70, 0x21u, *(_QWORD *)(v24 - 32), (_BYTE *)v25, (__int64)v68);
          v68[0] = *(_QWORD *)**a1;
          v52 = sub_B8C340(v68);
          v53 = v63;
          LOWORD(v53) = 0;
          v63 = v53;
          v54 = sub_F38250(v62, (__int64 *)(v24 + 24), v53, 0, v52, 0, 0, 0);
          sub_D5F1F0((__int64)v70, v54);
          v55 = sub_B6E160((__int64 *)**a1, 0x48u, 0, 0);
          v56 = 0;
          v69 = 257;
          if ( v55 )
            v56 = *(_QWORD *)(v55 + 24);
          v57 = sub_921880(v70, v56, v55, 0, 0, (__int64)v68, 0);
          v58 = v57;
          v68[0] = *(_QWORD *)(v24 + 48);
          if ( v68[0] )
          {
            sub_B96E90((__int64)v68, v68[0], 1);
            v59 = v58 + 48;
            if ( (__int64 *)(v58 + 48) == v68 )
            {
              if ( v68[0] )
                sub_B91220((__int64)v68, v68[0]);
              goto LABEL_76;
            }
            v60 = *(_QWORD *)(v58 + 48);
            if ( !v60 )
            {
LABEL_80:
              v61 = (unsigned __int8 *)v68[0];
              *(_QWORD *)(v58 + 48) = v68[0];
              if ( v61 )
                sub_B976B0((__int64)v68, v61, v59);
              goto LABEL_76;
            }
          }
          else
          {
            v59 = v57 + 48;
            if ( (__int64 *)(v57 + 48) == v68 || (v60 = *(_QWORD *)(v57 + 48)) == 0 )
            {
LABEL_76:
              v26 = dword_4FF8DA8;
              goto LABEL_34;
            }
          }
          v65 = v59;
          sub_B91220(v59, v60);
          v59 = v65;
          goto LABEL_80;
        }
LABEL_34:
        if ( v26 == 2 )
        {
          v68[0] = *(_QWORD *)**a1;
          v42 = sub_B8C320(v68);
          v43 = sub_29A5610(v24, v25, v42);
          v44 = v43;
          if ( *(_QWORD *)(v43 - 32) )
          {
            v45 = *(_QWORD **)(v43 - 16);
            v46 = *(_QWORD *)(v43 - 24);
            *v45 = v46;
            if ( v46 )
              *(_QWORD *)(v46 + 16) = *(_QWORD *)(v44 - 16);
          }
          *(_QWORD *)(v44 - 32) = v25;
          if ( v25 )
          {
            v47 = *(_QWORD *)(v25 + 16);
            *(_QWORD *)(v44 - 24) = v47;
            if ( v47 )
              *(_QWORD *)(v47 + 16) = v44 - 24;
            *(_QWORD *)(v44 - 16) = v25 + 16;
            *(_QWORD *)(v25 + 16) = v44 - 32;
          }
          sub_B99FD0(v44, 2u, 0);
          sub_B99FD0(v44, 0x17u, 0);
          sub_B99FD0(v24, 2u, 0);
          sub_B99FD0(v24, 0x17u, 0);
        }
        else
        {
          if ( *(_QWORD *)(v24 - 32) )
          {
            v27 = *(_QWORD *)(v24 - 24);
            **(_QWORD **)(v24 - 16) = v27;
            if ( v27 )
              *(_QWORD *)(v27 + 16) = *(_QWORD *)(v24 - 16);
          }
          *(_QWORD *)(v24 - 32) = v25;
          if ( v25 )
          {
            v28 = *(_QWORD *)(v25 + 16);
            *(_QWORD *)(v24 - 24) = v28;
            if ( v28 )
              *(_QWORD *)(v28 + 16) = v24 - 24;
            *(_QWORD *)(v24 - 16) = v25 + 16;
            *(_QWORD *)(v25 + 16) = v24 - 32;
          }
          sub_B99FD0(v24, 2u, 0);
          sub_B99FD0(v24, 0x17u, 0);
          if ( *(_QWORD *)(v24 - 32) && *(char *)(v24 + 7) < 0 )
          {
            v29 = sub_BD2BC0(v24);
            v31 = v29 + v30;
            if ( *(char *)(v24 + 7) < 0 )
              v31 -= sub_BD2BC0(v24);
            v32 = v31 >> 4;
            if ( (_DWORD)v32 )
            {
              v33 = 0;
              v34 = 16LL * (unsigned int)v32;
              while ( 1 )
              {
                v35 = 0;
                if ( *(char *)(v24 + 7) < 0 )
                  v35 = sub_BD2BC0(v24);
                if ( *(_DWORD *)(*(_QWORD *)(v35 + v33) + 8LL) == 7 )
                  break;
                v33 += 16;
                if ( v34 == v33 )
                  goto LABEL_55;
              }
              v36 = sub_B57640(v24, (__int64 *)7, v24 + 24, 0);
              sub_BD84D0(v24, v36);
              v39 = (__int64 **)*a1;
              v40 = *((unsigned int *)*a1 + 70);
              if ( v40 + 1 > (unsigned __int64)*((unsigned int *)*a1 + 71) )
              {
                sub_C8D5F0((__int64)(v39 + 34), v39 + 36, v40 + 1, 8u, v37, v38);
                v40 = *((unsigned int *)v39 + 70);
              }
              v39[34][v40] = v24;
              ++*((_DWORD *)v39 + 70);
            }
          }
        }
LABEL_55:
        v41 = *(_DWORD **)(v7 + 16);
        if ( v41 )
          --*v41;
        nullsub_61();
        v84 = &unk_49DA100;
        nullsub_63();
        if ( (_BYTE *)v70[0] == v71 )
          goto LABEL_7;
        _libc_free((unsigned __int64)v70[0]);
        v7 += 24;
        if ( v66 == v7 )
          break;
      }
      else
      {
LABEL_12:
        sub_C8CC70((__int64)(v8 + 22), v9, (__int64)a3, a4, a5, a6);
        if ( (_BYTE)a3 )
          goto LABEL_13;
LABEL_7:
        v7 += 24;
        if ( v66 == v7 )
          break;
      }
    }
  }
  if ( *(_BYTE *)(a2 + 25) || *(_QWORD *)(a2 + 40) != *(_QWORD *)(a2 + 32) )
  {
    *(_BYTE *)a1[2] = 1;
    result = *(_QWORD *)(a2 + 32);
    *(_BYTE *)(a2 + 24) = 1;
    if ( result != *(_QWORD *)(a2 + 40) )
      *(_QWORD *)(a2 + 40) = result;
  }
  else
  {
    *(_BYTE *)(a2 + 24) = 1;
    return a2;
  }
  return result;
}

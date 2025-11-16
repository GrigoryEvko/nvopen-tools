// Function: sub_2B45DE0
// Address: 0x2b45de0
//
unsigned __int64 __fastcall sub_2B45DE0(
        unsigned __int8 *a1,
        __int64 a2,
        void (__fastcall *a3)(__int64, unsigned __int64, unsigned __int8 *),
        __int64 a4,
        unsigned int a5)
{
  int v7; // r15d
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned __int64 v11; // r9
  char v12; // r10
  __int64 *v13; // r11
  unsigned __int64 v14; // rax
  unsigned int v15; // edx
  __int64 *v16; // r11
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // rdi
  _QWORD *v19; // rax
  unsigned __int64 *v20; // rax
  unsigned __int8 *v21; // rdx
  unsigned __int64 v23; // rbx
  unsigned int v24; // r12d
  unsigned int v25; // r13d
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // r12
  unsigned int v30; // r12d
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r11
  unsigned __int64 *v36; // r12
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rbx
  char *v41; // rsi
  int v42; // eax
  unsigned __int64 v43; // rax
  int v44; // edx
  unsigned int v45; // eax
  __int64 *v46; // [rsp+0h] [rbp-EA0h]
  unsigned __int64 v47; // [rsp+0h] [rbp-EA0h]
  __int64 *v48; // [rsp+8h] [rbp-E98h]
  __int64 *v49; // [rsp+8h] [rbp-E98h]
  unsigned int v50; // [rsp+10h] [rbp-E90h]
  int v52; // [rsp+18h] [rbp-E88h]
  unsigned __int64 *v53; // [rsp+18h] [rbp-E88h]
  unsigned __int64 v54; // [rsp+20h] [rbp-E80h] BYREF
  unsigned __int64 v55; // [rsp+28h] [rbp-E78h] BYREF
  unsigned __int64 v56; // [rsp+30h] [rbp-E70h] BYREF
  unsigned __int64 v57; // [rsp+38h] [rbp-E68h] BYREF
  unsigned __int64 v58; // [rsp+40h] [rbp-E60h] BYREF
  __int64 v59; // [rsp+48h] [rbp-E58h]
  char v60; // [rsp+50h] [rbp-E50h] BYREF
  unsigned __int64 v61; // [rsp+750h] [rbp-750h] BYREF
  unsigned __int8 *v62; // [rsp+758h] [rbp-748h]
  _QWORD v63[2]; // [rsp+760h] [rbp-740h] BYREF
  char v64; // [rsp+770h] [rbp-730h] BYREF

  v7 = *a1;
  v8 = *a1;
  v9 = (0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (unsigned __int64)sub_C64CA0) >> 47) ^ (0x9DDFEA08EB382D69LL * (_QWORD)sub_C64CA0))) >> 47;
  v55 = 0x9DDFEA08EB382D69LL
      * (v9
       ^ (0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (unsigned __int64)sub_C64CA0) >> 47) ^ (0x9DDFEA08EB382D69LL * (_QWORD)sub_C64CA0))));
  v54 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL
         * ((0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * (unsigned int)(v7 + 2)))
          ^ ((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * (unsigned int)(v7 + 2))) >> 47))) >> 47)
       ^ (0x9DDFEA08EB382D69LL
        * ((0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * (unsigned int)(v7 + 2)))
         ^ ((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * (unsigned int)(v7 + 2))) >> 47))));
  if ( (unsigned __int8)v7 <= 0x1Cu )
  {
    if ( !(unsigned __int8)sub_2B15E10((char *)a1, v8, v9, 0x9DDFEA08EB382D69LL, a5) )
      return v11;
    goto LABEL_16;
  }
  if ( (_BYTE)v7 != 61 )
  {
    if ( !(unsigned __int8)sub_2B15E10((char *)a1, v8, v9, 0x9DDFEA08EB382D69LL, a5) )
    {
      if ( ((unsigned int)(v7 - 42) <= 0x11 || (unsigned int)(v7 - 67) <= 0xC)
        && (unsigned __int8)(v8 - 51) > 1u
        && (unsigned int)(v7 - 48) > 1 )
      {
        v50 = v7 - 29;
        if ( v12 )
        {
          v46 = v13;
          v14 = sub_22AE640((unsigned int)(v7 - 42) <= 0x11);
          v15 = v7 - 42;
          v16 = v46;
          v54 = v14;
        }
        else
        {
          v49 = v13;
          v61 = sub_22AE640(v50);
          v43 = sub_C41E80((__int64 *)&v61, (__int64 *)&v54);
          v44 = *a1;
          v16 = v49;
          v54 = v43;
          LODWORD(v43) = v44 - 29;
          v15 = v44 - 42;
          v50 = v43;
        }
        v17 = *((_QWORD *)a1 + 1);
        if ( v15 > 0x11 )
          v18 = *(_QWORD *)(*((_QWORD *)a1 - 4) + 8LL);
        else
          v18 = *((_QWORD *)a1 + 1);
        v48 = v16;
        v47 = v17;
        v61 = sub_22AE640(v18);
        v58 = sub_22AE640(v47);
        v57 = sub_22AE640(v50);
        v55 = sub_22B2950((__int64 *)&v57, (__int64 *)&v58, (__int64 *)&v61);
        if ( (unsigned int)*a1 - 67 <= 0xC )
        {
          v19 = (_QWORD *)sub_986520((__int64)a1);
          v20 = (unsigned __int64 *)sub_2B45DE0(*v19, v48, a3, a4, 1);
          v62 = v21;
          v61 = (unsigned __int64)v20;
          v54 = sub_2B3B4C0((__int64 *)&v61, (__int64 *)&v54);
          v55 = sub_2B3B4C0((__int64 *)&v61, (__int64 *)&v55);
        }
        goto LABEL_23;
      }
      if ( (unsigned __int8)(v8 - 82) <= 1u )
      {
        v24 = *((_WORD *)a1 + 1) & 0x3F;
        if ( sub_B527F0((__int64)a1) )
        {
          v45 = sub_B52870(v24);
          if ( v24 > v45 )
            v24 = v45;
        }
        v25 = sub_B52F50(v24);
        v61 = sub_22AE640(*(_QWORD *)(*((_QWORD *)a1 - 8) + 8LL));
        v58 = sub_22AE640(v25);
        v57 = sub_22AE640(v24);
        v56 = sub_22AE640((unsigned int)*a1 - 29);
        v55 = sub_22B2710((__int64 *)&v56, (__int64 *)&v57, (__int64 *)&v58, (__int64 *)&v61);
        goto LABEL_23;
      }
      if ( (_BYTE)v8 == 85 )
      {
        v30 = sub_9B78C0((__int64)a1, v13);
        if ( (unsigned __int8)sub_9B7470(v30) )
        {
          v61 = sub_22AE640(v30);
          v58 = sub_22AE640((unsigned int)*a1 - 29);
        }
        else
        {
          v61 = sub_B43CA0((__int64)a1);
          v63[0] = &v64;
          v63[1] = 0x800000000LL;
          v62 = a1;
          sub_D39570((__int64)a1, (unsigned int *)v63);
          v58 = (unsigned __int64)&v60;
          v59 = 0x800000000LL;
          sub_D39570((__int64)a1, (unsigned int *)&v58);
          v52 = v59;
          sub_2B30DE0((__int64)&v58);
          sub_2B30DE0((__int64)v63);
          if ( !v52 )
          {
            v61 = sub_22AE640((unsigned __int64)a1);
            v54 = sub_C41E80((__int64 *)&v61, (__int64 *)&v54);
            v61 = sub_22AE640((unsigned __int64)a1);
            v58 = sub_22AE640((unsigned int)*a1 - 29);
            v55 = sub_C41E80((__int64 *)&v58, (__int64 *)&v61);
            goto LABEL_36;
          }
          v31 = *((_QWORD *)a1 - 4);
          v32 = 0;
          if ( v31 && !*(_BYTE *)v31 && *(_QWORD *)(v31 + 24) == *((_QWORD *)a1 + 10) )
            v32 = *((_QWORD *)a1 - 4);
          v61 = sub_22AE640(v32);
          v58 = sub_22AE640((unsigned int)*a1 - 29);
        }
        v55 = sub_C41E80((__int64 *)&v58, (__int64 *)&v61);
LABEL_36:
        if ( (a1[7] & 0x80u) != 0 )
        {
          v33 = sub_BD2BC0((__int64)a1);
          v53 = (unsigned __int64 *)(v34 + v33);
          v35 = (a1[7] & 0x80u) == 0 ? 0LL : sub_BD2BC0((__int64)a1);
          if ( v53 != (unsigned __int64 *)v35 )
          {
            v36 = (unsigned __int64 *)v35;
            do
            {
              v37 = *v36;
              v36 += 2;
              v38 = HIDWORD(v37);
              v39 = 0x9DDFEA08EB382D69LL * (HIDWORD(v37) ^ ((unsigned __int64)sub_C64CA0 + ((8 * v37) & 0x7FFFFFFF8LL)));
              v61 = 0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL * ((v39 >> 47) ^ v38 ^ v39)) >> 47)
                   ^ (0x9DDFEA08EB382D69LL * ((v39 >> 47) ^ v38 ^ v39)));
              v58 = 0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL
                     * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * *((unsigned int *)v36 - 1))) >> 47)
                      ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *((unsigned int *)v36 - 1))))) >> 47)
                   ^ (0x9DDFEA08EB382D69LL
                    * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * *((unsigned int *)v36 - 1))) >> 47)
                     ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *((unsigned int *)v36 - 1))))));
              v57 = 0x9DDFEA08EB382D69LL
                  * (((0x9DDFEA08EB382D69LL
                     * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * *((unsigned int *)v36 - 2))) >> 47)
                      ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *((unsigned int *)v36 - 2))))) >> 47)
                   ^ (0x9DDFEA08EB382D69LL
                    * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)sub_C64CA0 + 8 * *((unsigned int *)v36 - 2))) >> 47)
                     ^ (0x9DDFEA08EB382D69LL * ((_QWORD)sub_C64CA0 + 8 * *((unsigned int *)v36 - 2))))));
              v55 = sub_22B2710((__int64 *)&v57, (__int64 *)&v58, (__int64 *)&v61, (__int64 *)&v55);
            }
            while ( v53 != v36 );
          }
        }
        goto LABEL_23;
      }
      if ( (_BYTE)v8 == 63 )
      {
        if ( (*((_DWORD *)a1 + 1) & 0x7FFFFFF) == 2 && **((_BYTE **)a1 - 4) == 17 )
        {
          v55 = sub_22AE640(*((_QWORD *)a1 - 8));
          goto LABEL_23;
        }
      }
      else if ( (unsigned __int8)(v8 - 51) > 1u && (unsigned int)(v7 - 48) > 1
             || **(_BYTE **)(sub_986520((__int64)a1) + 32) == 17 )
      {
        v55 = sub_22AE640((unsigned int)(v7 - 29));
        goto LABEL_23;
      }
      v55 = sub_22AE640((unsigned __int64)a1);
LABEL_23:
      v26 = *((_QWORD *)a1 + 5);
      v27 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * (HIDWORD(v26) ^ ((unsigned __int64)sub_C64CA0 + ((8 * v26) & 0x7FFFFFFF8LL))))
           ^ HIDWORD(v26)
           ^ ((0x9DDFEA08EB382D69LL * (HIDWORD(v26) ^ ((unsigned __int64)sub_C64CA0 + ((8 * v26) & 0x7FFFFFFF8LL)))) >> 47));
      v61 = 0x9DDFEA08EB382D69LL * ((v27 >> 47) ^ v27);
      return sub_C41E80((__int64 *)&v61, (__int64 *)&v54);
    }
    if ( (_BYTE)v8 == 90 )
    {
      v40 = v10 * (((unsigned __int64)(v10 * ((_QWORD)&loc_C64D07 + 1)) >> 47) ^ (v10 * ((_QWORD)&loc_C64D07 + 1)));
      v54 = ((v40 >> 47) ^ v40) * v10;
LABEL_44:
      v41 = (char *)*((_QWORD *)a1 - 8);
      v58 = 1;
      sub_2B25A00(&v61, v41, &v58);
      if ( (unsigned __int8)sub_2B0D9E0(v61) || (v42 = **((unsigned __int8 **)a1 - 4), v42 == 12) || v42 == 13 )
      {
        sub_228BF40((unsigned __int64 **)&v61);
        sub_228BF40((unsigned __int64 **)&v58);
        return v54;
      }
      else
      {
        sub_228BF40((unsigned __int64 **)&v61);
        sub_228BF40((unsigned __int64 **)&v58);
        sub_22AE640(*((_QWORD *)a1 - 8));
        return v54;
      }
    }
LABEL_16:
    if ( (unsigned int)(v7 - 12) > 1 )
      return v11;
    v23 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * ((unsigned __int64)&loc_C64D07 + 1)) >> 47)
         ^ (0x9DDFEA08EB382D69LL * ((_QWORD)&loc_C64D07 + 1)));
    v11 = 0x9DDFEA08EB382D69LL * ((v23 >> 47) ^ v23);
    v54 = v11;
    if ( (_BYTE)v8 != 90 )
      return v11;
    goto LABEL_44;
  }
  v28 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (unsigned __int64)&loc_C64DA0) >> 47) ^ (0x9DDFEA08EB382D69LL * (_QWORD)&loc_C64DA0));
  v61 = *((_QWORD *)a1 + 1);
  v58 = 0x9DDFEA08EB382D69LL * ((v28 >> 47) ^ v28);
  v54 = sub_2B3B430((__int64 *)&v61, (__int64 *)&v58, (__int64 *)&v54);
  v29 = v54;
  if ( sub_B46500(a1) || (a1[2] & 1) != 0 )
    return sub_22AE640((unsigned __int64)a1);
  a3(a4, v29, a1);
  return v54;
}

// Function: sub_2AD5FD0
// Address: 0x2ad5fd0
//
_QWORD *__fastcall sub_2AD5FD0(__int64 a1, unsigned __int8 *a2, __int64 *a3, unsigned __int64 a4, __int64 a5)
{
  _QWORD *result; // rax
  unsigned __int8 v10; // al
  __int64 v11; // rax
  unsigned __int8 **v12; // rsi
  unsigned __int8 **v13; // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  unsigned int v16; // ecx
  unsigned __int8 **v17; // rdx
  unsigned __int8 *v18; // r10
  __int64 v19; // rdi
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // r8
  unsigned int v23; // esi
  unsigned __int8 **v24; // rdx
  unsigned __int8 *v25; // r11
  __int64 v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rcx
  int v29; // eax
  __int64 v30; // rax
  char v31; // r13
  __int64 v32; // rax
  __int64 v33; // rdx
  void *v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  int v37; // eax
  __int64 v38; // r9
  char *v39; // rdx
  __int64 v40; // r9
  int v41; // edx
  int v42; // ecx
  int v43; // edx
  int v44; // r11d
  __int64 v45; // r14
  __int64 v46; // r13
  int v47; // r15d
  __int64 v48; // r8
  __int64 v49; // [rsp+8h] [rbp-68h]
  __int64 v50; // [rsp+8h] [rbp-68h]
  __int64 v51; // [rsp+10h] [rbp-60h]
  unsigned __int8 v52; // [rsp+10h] [rbp-60h]
  char v53; // [rsp+18h] [rbp-58h]
  __int64 v54; // [rsp+18h] [rbp-58h]
  int v55; // [rsp+18h] [rbp-58h]
  __int64 v56; // [rsp+18h] [rbp-58h]
  _QWORD *v57; // [rsp+18h] [rbp-58h]
  _QWORD *v58; // [rsp+18h] [rbp-58h]
  _QWORD *v59; // [rsp+18h] [rbp-58h]
  __int64 v60[10]; // [rsp+20h] [rbp-50h] BYREF

  if ( *a2 != 84 )
  {
    if ( *a2 == 67 )
    {
      v54 = a5;
      result = (_QWORD *)sub_2AC4440((__int64 *)a1, (__int64)a2, (__int64)a3, a4, a5);
      a5 = v54;
      if ( result )
        return result;
    }
    v51 = a5;
    v60[3] = (__int64)sub_2AA76A0;
    v60[2] = (__int64)sub_2AA76B0;
    v53 = sub_2BF1270(v60, a5);
    sub_A17130((__int64)v60);
    if ( !v53 )
    {
      v10 = *a2;
      if ( *a2 == 85 )
        return (_QWORD *)sub_2AC4ED0(a1, a2, a3, a4, v51);
      if ( v10 == 62 )
      {
        v11 = *(_QWORD *)(a1 + 32);
        v12 = *(unsigned __int8 ***)(v11 + 536);
        v13 = &v12[3 * *(unsigned int *)(v11 + 544)];
        if ( v12 != v13 )
        {
          while ( a2 != *v12 && a2 != v12[1] && a2 != v12[2] )
          {
            v12 += 3;
            if ( v13 == v12 )
              return (_QWORD *)sub_2AB6F90(a1, (__int64)a2, a3, a4, v51);
          }
          return (_QWORD *)sub_2AC5A70(a1, (__int64)v12, (__int64)a3);
        }
        return (_QWORD *)sub_2AB6F90(a1, (__int64)a2, a3, a4, v51);
      }
      if ( (unsigned __int8)(v10 - 61) <= 1u )
        return (_QWORD *)sub_2AB6F90(a1, (__int64)a2, a3, a4, v51);
      v14 = *(unsigned int *)(a1 + 232);
      v15 = *(_QWORD *)(a1 + 216);
      if ( (_DWORD)v14 )
      {
        v16 = (v14 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v17 = (unsigned __int8 **)(v15 + 16LL * v16);
        v18 = *v17;
        if ( a2 == *v17 )
        {
LABEL_23:
          if ( v17 != (unsigned __int8 **)(v15 + 16 * v14) )
            return (_QWORD *)sub_2AC5550((__int64 *)a1, a2, a3);
        }
        else
        {
          v43 = 1;
          while ( v18 != (unsigned __int8 *)-4096LL )
          {
            v44 = v43 + 1;
            v16 = (v14 - 1) & (v43 + v16);
            v17 = (unsigned __int8 **)(v15 + 16LL * v16);
            v18 = *v17;
            if ( a2 == *v17 )
              goto LABEL_23;
            v43 = v44;
          }
        }
      }
      if ( (unsigned __int8)sub_2AB77E0(a1, (__int64)a2, v51) )
      {
        v37 = *a2;
        if ( (_BYTE)v37 == 63 )
        {
          result = (_QWORD *)sub_22077B0(0xA0u);
          if ( !result )
            return result;
          v58 = result;
          sub_2AC1B80((__int64)result, 17, a3, &a3[a4], a2, v38);
          v39 = (char *)&unk_4A241B8;
        }
        else
        {
          if ( (_BYTE)v37 != 86 )
          {
            if ( (unsigned int)(v37 - 67) > 0xC )
              return (_QWORD *)sub_2AC5DB0((__int64 *)a1, a2, (__int64)a3, a4);
            v45 = *a3;
            v46 = *((_QWORD *)a2 + 1);
            v47 = v37 - 29;
            result = (_QWORD *)sub_22077B0(0xB0u);
            if ( result )
            {
              v59 = result;
              sub_2ABA9E0((__int64)result, 16, v45, a2, v48);
              *((_DWORD *)v59 + 40) = v47;
              *v59 = &unk_4A23F58;
              v59[5] = &unk_4A23F90;
              v59[12] = &unk_4A23FC8;
              v59[21] = v46;
              return v59;
            }
            return result;
          }
          result = (_QWORD *)sub_22077B0(0xA0u);
          if ( !result )
            return result;
          v58 = result;
          sub_2AC1B80((__int64)result, 24, a3, &a3[a4], a2, v40);
          v39 = (char *)&unk_4A23E20;
        }
        *v58 = v39 + 16;
        v58[5] = v39 + 80;
        v58[12] = v39 + 136;
        return v58;
      }
    }
    return 0;
  }
  if ( **(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL) != *((_QWORD *)a2 + 5) )
    return sub_2AD5B30(a1, (__int64)a2, (__int64)a3);
  result = (_QWORD *)sub_2ABB7D0((__int64 *)a1, (__int64)a2, a3, a4, a5);
  if ( !result )
  {
    v19 = *(_QWORD *)(a1 + 32);
    v20 = *a3;
    v21 = *(unsigned int *)(v19 + 104);
    v22 = *(_QWORD *)(v19 + 88);
    if ( (_DWORD)v21 )
    {
      v23 = (v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = (unsigned __int8 **)(v22 + 16LL * (((_DWORD)v21 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4))));
      v25 = *v24;
      if ( a2 == *v24 )
      {
LABEL_27:
        if ( v24 != (unsigned __int8 **)(v22 + 16 * v21) )
        {
          v49 = sub_2AA8FC0(v19 + 80, (__int64)a2);
          v26 = v49 + 8;
          v60[0] = sub_2AB16F0(a1, *(_QWORD *)(v49 + 40));
          v29 = 1;
          if ( BYTE4(v60[0]) )
            v29 = v60[0];
          v55 = v29;
          v52 = sub_B19060(*(_QWORD *)(a1 + 40) + 256LL, (__int64)a2, v27, v28);
          v30 = *(_QWORD *)(a1 + 40);
          v31 = 0;
          if ( !(unsigned __int8)sub_31A4BE0(*(_QWORD *)(v30 + 496)) )
            v31 = *(_BYTE *)(v49 + 73);
          v32 = sub_22077B0(0xA8u);
          if ( v32 )
          {
            v50 = v32;
            v60[0] = 0;
            sub_2AAFAC0(v32, 36, (__int64)a2, v20, v60, v36);
            sub_9C6650(v60);
            v32 = v50;
            *(_QWORD *)(v50 + 152) = v26;
            *(_QWORD *)v50 = &unk_4A24C80;
            v34 = &unk_4A24CD0;
            *(_QWORD *)(v50 + 96) = &unk_4A24D08;
            v33 = v52;
            *(_QWORD *)(v50 + 40) = &unk_4A24CD0;
            *(_BYTE *)(v50 + 160) = v52;
            *(_BYTE *)(v50 + 161) = v31;
            *(_DWORD *)(v50 + 164) = v55;
          }
LABEL_34:
          v56 = v32;
          sub_2AAECA0(v32 + 40, a3[1], v33, (__int64)v34, v35, v36);
          return (_QWORD *)v56;
        }
      }
      else
      {
        v41 = 1;
        while ( v25 != (unsigned __int8 *)-4096LL )
        {
          v42 = v41 + 1;
          v23 = (v21 - 1) & (v41 + v23);
          v24 = (unsigned __int8 **)(v22 + 16LL * v23);
          v25 = *v24;
          if ( a2 == *v24 )
            goto LABEL_27;
          v41 = v42;
        }
      }
    }
    v32 = sub_22077B0(0x98u);
    if ( v32 )
    {
      v57 = (_QWORD *)v32;
      v60[0] = 0;
      sub_2AAFAC0(v32, 32, (__int64)a2, v20, v60, v36);
      sub_9C6650(v60);
      v32 = (__int64)v57;
      *v57 = &unk_4A24BD8;
      v34 = &unk_4A24C28;
      v33 = (__int64)&unk_4A24C60;
      v57[5] = &unk_4A24C28;
      v57[12] = &unk_4A24C60;
    }
    goto LABEL_34;
  }
  return result;
}

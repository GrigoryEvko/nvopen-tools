// Function: sub_2E5F1F0
// Address: 0x2e5f1f0
//
__int64 __fastcall sub_2E5F1F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  __int64 (*v5)(); // rax
  __int64 v6; // rbx
  unsigned int v7; // r15d
  unsigned __int8 v8; // al
  __int64 *v9; // r13
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 *v13; // r12
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rax
  _QWORD *v17; // rdi
  _QWORD *v18; // rsi
  bool v19; // al
  __int64 (*v21)(); // r12
  __int64 v22; // rax
  __int64 (*v23)(); // rax
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 *v26; // r8
  int v27; // ecx
  unsigned int v28; // edx
  __int64 *v29; // rdi
  __int64 v30; // r9
  int v31; // edi
  int v32; // r10d
  __int64 v33; // [rsp+0h] [rbp-70h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  __int64 *v36; // [rsp+18h] [rbp-58h]
  __int64 v37; // [rsp+20h] [rbp-50h]
  __int64 v38; // [rsp+28h] [rbp-48h]
  __int64 v39[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 32LL);
  v4 = *(_QWORD *)(v3 + 16);
  v37 = *(_QWORD *)(v3 + 32);
  v34 = 0;
  v33 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 200LL))(v4);
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 128LL);
  if ( v5 != sub_2DAC790 )
    v34 = ((__int64 (__fastcall *)(__int64))v5)(v4);
  v6 = *(_QWORD *)(a2 + 32);
  v38 = v6 + 40LL * (*(_DWORD *)(a2 + 40) & 0xFFFFFF);
  if ( v38 == v6 )
    return 1;
  while ( 1 )
  {
    if ( *(_BYTE *)v6 )
      goto LABEL_21;
    v7 = *(_DWORD *)(v6 + 8);
    if ( !v7 )
      goto LABEL_21;
    if ( v7 - 1 <= 0x3FFFFFFE )
      break;
LABEL_17:
    if ( (*(_BYTE *)(v6 + 3) & 0x10) == 0 )
    {
      v14 = sub_2EBEE10(v37, v7);
      v15 = *(_DWORD *)(a1 + 72);
      v16 = *(_QWORD *)(v14 + 24);
      v39[0] = v16;
      if ( v15 )
      {
        v24 = *(_QWORD *)(a1 + 64);
        v25 = *(unsigned int *)(a1 + 80);
        v26 = (__int64 *)(v24 + 8 * v25);
        if ( !(_DWORD)v25 )
          goto LABEL_21;
        v27 = v25 - 1;
        v28 = v27 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v29 = (__int64 *)(v24 + 8LL * v28);
        v30 = *v29;
        if ( v16 != *v29 )
        {
          v31 = 1;
          while ( v30 != -4096 )
          {
            v32 = v31 + 1;
            v28 = v27 & (v31 + v28);
            v29 = (__int64 *)(v24 + 8LL * v28);
            v30 = *v29;
            if ( v16 == *v29 )
              goto LABEL_29;
            v31 = v32;
          }
          goto LABEL_21;
        }
LABEL_29:
        v19 = v26 != v29;
      }
      else
      {
        v17 = *(_QWORD **)(a1 + 88);
        v18 = &v17[*(unsigned int *)(a1 + 96)];
        v19 = v18 != sub_2E5D7F0(v17, (__int64)v18, v39);
      }
      if ( v19 )
        return 0;
    }
LABEL_21:
    v6 += 40;
    if ( v38 == v6 )
      return 1;
  }
  v8 = *(_BYTE *)(v6 + 3);
  if ( (v8 & 0x10) != 0 )
  {
    if ( (((v8 & 0x10) != 0) & (v8 >> 6)) == 0 )
      return 0;
    v9 = *(__int64 **)(a1 + 8);
    v10 = 8LL * *(unsigned int *)(a1 + 16);
    v36 = &v9[(unsigned __int64)v10 / 8];
    v11 = v10 >> 3;
    v12 = v10 >> 5;
    if ( v12 )
    {
      v13 = &v9[4 * v12];
      while ( !(unsigned __int8)sub_2E31DD0(*v9, v7, -1, -1) )
      {
        if ( (unsigned __int8)sub_2E31DD0(v9[1], v7, -1, -1) )
        {
          ++v9;
          break;
        }
        if ( (unsigned __int8)sub_2E31DD0(v9[2], v7, -1, -1) )
        {
          v9 += 2;
          break;
        }
        if ( (unsigned __int8)sub_2E31DD0(v9[3], v7, -1, -1) )
        {
          v9 += 3;
          break;
        }
        v9 += 4;
        if ( v13 == v9 )
        {
          v11 = v36 - v9;
          goto LABEL_31;
        }
      }
LABEL_16:
      if ( v36 != v9 )
        return 0;
      goto LABEL_17;
    }
LABEL_31:
    if ( v11 != 2 )
    {
      if ( v11 != 3 )
      {
        if ( v11 != 1 )
          goto LABEL_17;
        goto LABEL_34;
      }
      if ( (unsigned __int8)sub_2E31DD0(*v9, v7, -1, -1) )
        goto LABEL_16;
      ++v9;
    }
    if ( (unsigned __int8)sub_2E31DD0(*v9, v7, -1, -1) )
      goto LABEL_16;
    ++v9;
LABEL_34:
    if ( !(unsigned __int8)sub_2E31DD0(*v9, v7, -1, -1) )
      goto LABEL_17;
    goto LABEL_16;
  }
  if ( (unsigned __int8)sub_2EBF3A0(v37, v7) )
    goto LABEL_21;
  v21 = *(__int64 (**)())(*(_QWORD *)v33 + 200LL);
  v22 = sub_2E88D60(a2, v7);
  if ( v21 != sub_2E4EE50 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64))v21)(v33, v7, v22) )
      goto LABEL_21;
  }
  v23 = *(__int64 (**)())(*(_QWORD *)v34 + 32LL);
  if ( v23 != sub_2E4EE60 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, __int64))v23)(v34, v6) )
      goto LABEL_21;
  }
  return 0;
}

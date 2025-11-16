// Function: sub_35144C0
// Address: 0x35144c0
//
__int64 __fastcall sub_35144C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  _QWORD *v10; // rax
  __int64 *v11; // r13
  __int64 v12; // r11
  int v13; // r10d
  __int64 v14; // r8
  __int64 *v15; // rdx
  unsigned int v16; // edi
  __int64 *v17; // rax
  __int64 v18; // rcx
  __int64 v19; // r8
  bool v20; // al
  _QWORD *v21; // rdi
  _QWORD *v22; // rsi
  unsigned int v23; // r14d
  unsigned __int64 v24; // r12
  unsigned __int64 v25; // rax
  __int64 v26; // r12
  unsigned int v27; // esi
  int v28; // r9d
  int v29; // r9d
  __int64 v30; // rdi
  unsigned int v31; // ecx
  int v32; // eax
  __int64 v33; // rsi
  int v34; // eax
  int v35; // ecx
  int v36; // ecx
  __int64 v37; // rdi
  __int64 *v38; // r9
  unsigned int v39; // r14d
  int v40; // r10d
  __int64 v41; // rsi
  int v42; // edx
  __int64 v43; // rcx
  int v44; // edx
  unsigned int v45; // eax
  __int64 v46; // rsi
  int v47; // edi
  __int64 *v48; // rax
  __int64 v49; // r14
  __int64 v50; // r13
  int v51; // r14d
  __int64 *v52; // r10
  __int64 v53; // [rsp+0h] [rbp-B0h]
  unsigned int v54; // [rsp+Ch] [rbp-A4h]
  __int64 v56; // [rsp+10h] [rbp-A0h]
  __int64 v57; // [rsp+10h] [rbp-A0h]
  __int64 v58; // [rsp+10h] [rbp-A0h]
  __int64 *v62; // [rsp+38h] [rbp-78h]
  unsigned __int64 v63; // [rsp+48h] [rbp-68h] BYREF
  __int64 v64; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int64 v65; // [rsp+58h] [rbp-58h] BYREF
  unsigned __int64 v66[2]; // [rsp+60h] [rbp-50h] BYREF
  char v67; // [rsp+70h] [rbp-40h]

  result = 0;
  if ( *(_DWORD *)(a4 + 56) )
  {
    sub_B2EE70((__int64)v66, **(_QWORD **)(a2 + 32), 0);
    if ( v67 )
    {
      if ( *(_DWORD *)(a2 + 120) == 2 )
      {
        v48 = *(__int64 **)(a2 + 112);
        v49 = *v48;
        v50 = v48[1];
        if ( sub_2E322C0(*v48, v50) || sub_2E322C0(v50, v49) )
        {
          sub_F02DB0(v66, 2 * LODWORD(qword_501F228[8]), 0x96u);
          v54 = v66[0];
          goto LABEL_6;
        }
      }
      v10 = &qword_501F1E0;
    }
    else
    {
      v10 = &qword_501F2C0;
    }
    sub_F02DB0(v66, *((_DWORD *)v10 + 34), 0x64u);
    v54 = v66[0];
LABEL_6:
    v66[0] = sub_2F06CB0(*(_QWORD *)(a1 + 536), a2);
    v63 = sub_1098D20(v66, a5);
    v11 = *(__int64 **)(a3 + 64);
    v53 = a1 + 888;
    if ( v11 == &v11[*(unsigned int *)(a3 + 72)] )
      return 0;
    v62 = &v11[*(unsigned int *)(a3 + 72)];
    v12 = a3;
    while ( 1 )
    {
      v26 = *v11;
      v27 = *(_DWORD *)(a1 + 912);
      v64 = *v11;
      if ( v27 )
      {
        v13 = 1;
        v14 = *(_QWORD *)(a1 + 896);
        v15 = 0;
        v16 = (v27 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v17 = (__int64 *)(v14 + 16LL * v16);
        v18 = *v17;
        if ( v26 == *v17 )
        {
LABEL_9:
          v19 = v17[1];
          v20 = a4 == v19;
          goto LABEL_10;
        }
        while ( v18 != -4096 )
        {
          if ( v18 == -8192 && !v15 )
            v15 = v17;
          v16 = (v27 - 1) & (v13 + v16);
          v17 = (__int64 *)(v14 + 16LL * v16);
          v18 = *v17;
          if ( v26 == *v17 )
            goto LABEL_9;
          ++v13;
        }
        if ( !v15 )
          v15 = v17;
        v34 = *(_DWORD *)(a1 + 904);
        ++*(_QWORD *)(a1 + 888);
        v32 = v34 + 1;
        if ( 4 * v32 < 3 * v27 )
        {
          if ( v27 - *(_DWORD *)(a1 + 908) - v32 <= v27 >> 3 )
          {
            v58 = v12;
            sub_3512300(v53, v27);
            v35 = *(_DWORD *)(a1 + 912);
            if ( !v35 )
            {
LABEL_70:
              ++*(_DWORD *)(a1 + 904);
              BUG();
            }
            v36 = v35 - 1;
            v37 = *(_QWORD *)(a1 + 896);
            v38 = 0;
            v39 = v36 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v12 = v58;
            v40 = 1;
            v32 = *(_DWORD *)(a1 + 904) + 1;
            v15 = (__int64 *)(v37 + 16LL * v39);
            v41 = *v15;
            if ( v26 != *v15 )
            {
              while ( v41 != -4096 )
              {
                if ( v41 == -8192 && !v38 )
                  v38 = v15;
                v39 = v36 & (v40 + v39);
                v15 = (__int64 *)(v37 + 16LL * v39);
                v41 = *v15;
                if ( v26 == *v15 )
                  goto LABEL_24;
                ++v40;
              }
              if ( v38 )
                v15 = v38;
            }
          }
          goto LABEL_24;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 888);
      }
      v57 = v12;
      sub_3512300(v53, 2 * v27);
      v28 = *(_DWORD *)(a1 + 912);
      if ( !v28 )
        goto LABEL_70;
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 896);
      v12 = v57;
      v31 = v29 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
      v32 = *(_DWORD *)(a1 + 904) + 1;
      v15 = (__int64 *)(v30 + 16LL * v31);
      v33 = *v15;
      if ( v26 != *v15 )
      {
        v51 = 1;
        v52 = 0;
        while ( v33 != -4096 )
        {
          if ( v33 == -8192 && !v52 )
            v52 = v15;
          v31 = v29 & (v51 + v31);
          v15 = (__int64 *)(v30 + 16LL * v31);
          v33 = *v15;
          if ( v26 == *v15 )
            goto LABEL_24;
          ++v51;
        }
        if ( v52 )
          v15 = v52;
      }
LABEL_24:
      *(_DWORD *)(a1 + 904) = v32;
      if ( *v15 != -4096 )
        --*(_DWORD *)(a1 + 908);
      *v15 = v26;
      v20 = 0;
      v19 = 0;
      v15[1] = 0;
LABEL_10:
      if ( v12 == v26 || v20 )
        goto LABEL_19;
      if ( a7 )
      {
        if ( *(_DWORD *)(a7 + 16) )
        {
          v42 = *(_DWORD *)(a7 + 24);
          v43 = *(_QWORD *)(a7 + 8);
          if ( !v42 )
            goto LABEL_19;
          v44 = v42 - 1;
          v45 = v44 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v46 = *(_QWORD *)(v43 + 8LL * v45);
          if ( v26 != v46 )
          {
            v47 = 1;
            while ( v46 != -4096 )
            {
              v45 = v44 & (v47 + v45);
              v46 = *(_QWORD *)(v43 + 8LL * v45);
              if ( v26 == v46 )
                goto LABEL_15;
              ++v47;
            }
            goto LABEL_19;
          }
        }
        else
        {
          v21 = *(_QWORD **)(a7 + 32);
          v22 = &v21[*(unsigned int *)(a7 + 40)];
          if ( v22 == sub_3510810(v21, (__int64)v22, &v64) )
            goto LABEL_19;
        }
      }
LABEL_15:
      if ( a6 != v19 && *(_QWORD *)(*(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 8) - 8) == v26 && a2 != v26 )
      {
        v56 = v12;
        v23 = sub_2E441D0(*(_QWORD *)(a1 + 528), v26, v12);
        v66[0] = sub_2F06CB0(*(_QWORD *)(a1 + 536), v26);
        v65 = sub_1098D20(v66, v23);
        v24 = sub_1098D20(&v63, 0x80000000 - v54);
        v25 = sub_1098D20(&v65, v54);
        v12 = v56;
        if ( v24 <= v25 )
          return 1;
      }
LABEL_19:
      if ( v62 == ++v11 )
        return 0;
    }
  }
  return result;
}

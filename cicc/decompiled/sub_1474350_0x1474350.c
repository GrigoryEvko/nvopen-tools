// Function: sub_1474350
// Address: 0x1474350
//
__int64 __fastcall sub_1474350(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  unsigned int v9; // r10d
  __int64 v11; // rax
  __int64 v12; // rax
  char v13; // al
  __int64 *v14; // rax
  __int64 v15; // rax
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // al
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rdi
  unsigned int v27; // ecx
  __int64 *v28; // rdx
  __int64 v29; // r9
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r15
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // r8
  int v40; // edx
  int v41; // r10d
  __int64 v42; // [rsp+8h] [rbp-68h]
  __int64 v43; // [rsp+10h] [rbp-60h]
  __int64 i; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+18h] [rbp-58h]
  __int64 j; // [rsp+18h] [rbp-58h]
  __int64 v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+20h] [rbp-50h]
  __int64 v50; // [rsp+20h] [rbp-50h]
  _QWORD *v51; // [rsp+20h] [rbp-50h]
  __int64 v52; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  __int64 v55; // [rsp+28h] [rbp-48h]
  _QWORD v56[8]; // [rsp+30h] [rbp-40h] BYREF

  if ( a2 && !(unsigned __int8)sub_1481140(a1, a3, a4, a5) )
  {
    v11 = sub_13FCB50(a2);
    v9 = 0;
    v46 = v11;
    if ( !v11 )
      return v9;
    v12 = sub_157EBA0(v11);
    v9 = 0;
    if ( *(_BYTE *)(v12 + 16) != 26
      || (*(_DWORD *)(v12 + 20) & 0xFFFFFFF) != 3
      || (v13 = sub_148B0D0(a1, a3, a4, a5, *(_QWORD *)(v12 - 72), **(_QWORD **)(a2 + 32) != *(_QWORD *)(v12 - 24)),
          v9 = 0,
          !v13) )
    {
      if ( *(_BYTE *)(a1 + 488) )
        return v9;
      *(_BYTE *)(a1 + 488) = 1;
      v14 = sub_1473850(a1, a2);
      v48 = sub_1457040((__int64)v14, v46, a1);
      if ( v48 != sub_1456E90(a1) )
      {
        v42 = v48;
        v43 = sub_1456040(v48);
        v49 = sub_145CF80(a1, v43, 1, 0);
        v15 = sub_145CF80(a1, v43, 0, 0);
        v16 = sub_14799E0(a1, v15, v49, a2, 3);
        if ( (unsigned __int8)sub_148AAB0(a1, a3, a4, a5, 36, v16, v42) )
          goto LABEL_33;
      }
      v17 = *(_QWORD *)(a1 + 48);
      if ( !*(_BYTE *)(v17 + 184) )
      {
        v52 = *(_QWORD *)(a1 + 48);
        sub_14CDF70(v52);
        v17 = v52;
      }
      v18 = *(_QWORD *)(v17 + 8);
      for ( i = v18 + 32LL * *(unsigned int *)(v17 + 16); i != v18; v18 += 32 )
      {
        v19 = *(_QWORD *)(v18 + 16);
        if ( v19 )
        {
          v50 = v18;
          v20 = sub_157EBA0(v46);
          v21 = sub_15CCEE0(*(_QWORD *)(a1 + 56), v19, v20);
          v18 = v50;
          if ( v21 )
          {
            v22 = sub_148B0D0(a1, a3, a4, a5, *(_QWORD *)(v19 - 24LL * (*(_DWORD *)(v19 + 20) & 0xFFFFFFF)), 0);
            v18 = v50;
            v9 = v22;
            if ( (_BYTE)v22 )
              goto LABEL_34;
          }
        }
      }
      v23 = *(_QWORD *)(a1 + 56);
      v24 = *(unsigned int *)(v23 + 48);
      if ( (_DWORD)v24 )
      {
        v25 = *(_QWORD *)(v23 + 32);
        v26 = **(_QWORD **)(a2 + 32);
        v27 = (v24 - 1) & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
        v28 = (__int64 *)(v25 + 16LL * v27);
        v29 = *v28;
        if ( v26 == *v28 )
        {
LABEL_21:
          if ( v28 != (__int64 *)(v25 + 16 * v24) && v28[1] )
          {
            if ( (unsigned __int8)sub_148B340(a1, v46, a3, a4, a5) )
            {
LABEL_33:
              v9 = 1;
LABEL_34:
              *(_BYTE *)(a1 + 488) = 0;
              return v9;
            }
            v51 = (_QWORD *)sub_15CC510(*(_QWORD *)(a1 + 56), v46, v30, v31, v32);
            for ( j = sub_15CC510(*(_QWORD *)(a1 + 56), **(_QWORD **)(a2 + 32), v33, v34, v35);
                  (_QWORD *)j != v51;
                  v51 = (_QWORD *)v51[1] )
            {
              v36 = *v51;
              if ( (unsigned __int8)sub_148B340(a1, *v51, a3, a4, a5) )
                goto LABEL_33;
              v37 = sub_157F0B0(v36);
              if ( v37 )
              {
                v54 = v37;
                v38 = sub_157EBA0(v37);
                if ( *(_BYTE *)(v38 + 16) == 26 && (*(_DWORD *)(v38 + 20) & 0xFFFFFFF) == 3 )
                {
                  v39 = *(_QWORD *)(v38 - 72);
                  v45 = v38;
                  v56[0] = v54;
                  v55 = v39;
                  v56[1] = v36;
                  if ( (unsigned __int8)sub_15CC350(v56) )
                  {
                    if ( (unsigned __int8)sub_148B0D0(a1, a3, a4, a5, v55, *(_QWORD *)(v45 - 24) != v36) )
                      goto LABEL_33;
                  }
                }
              }
            }
          }
        }
        else
        {
          v40 = 1;
          while ( v29 != -8 )
          {
            v41 = v40 + 1;
            v27 = (v24 - 1) & (v40 + v27);
            v28 = (__int64 *)(v25 + 16LL * v27);
            v29 = *v28;
            if ( v26 == *v28 )
              goto LABEL_21;
            v40 = v41;
          }
        }
      }
      v9 = 0;
      goto LABEL_34;
    }
  }
  return 1;
}

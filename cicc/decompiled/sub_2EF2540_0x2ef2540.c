// Function: sub_2EF2540
// Address: 0x2ef2540
//
void __fastcall sub_2EF2540(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 *v8; // rax
  __int64 *i; // rbx
  __int64 v10; // r12
  __int64 *v11; // rsi
  __int64 v12; // rax
  __int64 v13; // r10
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  unsigned __int64 v17; // rdx
  char *v18; // rsi
  __int64 v19; // r11
  unsigned __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // rsi
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r9
  unsigned int v26; // edi
  int v27; // r8d
  __int16 *v28; // rdi
  int v29; // r10d
  __int64 v30; // rsi
  __int64 v31; // rdi
  char v32; // di
  __int64 v33; // rbx
  _QWORD *v34; // rdx
  __int64 v35; // rsi
  _QWORD *v36; // rdi
  _QWORD *v37; // rdi
  __int64 v38; // [rsp+0h] [rbp-80h]
  char v39; // [rsp+Fh] [rbp-71h]
  __int64 v40; // [rsp+10h] [rbp-70h]
  __int64 v42; // [rsp+20h] [rbp-60h]
  char v43; // [rsp+20h] [rbp-60h]
  __int64 *v44; // [rsp+28h] [rbp-58h]
  unsigned __int64 v45; // [rsp+30h] [rbp-50h]
  unsigned __int64 v46; // [rsp+30h] [rbp-50h]
  __int64 v47; // [rsp+30h] [rbp-50h]
  __int64 v49[7]; // [rsp+48h] [rbp-38h] BYREF

  v8 = *(__int64 **)(a2 + 64);
  v38 = a5 | a4;
  v44 = &v8[*(unsigned int *)(a2 + 72)];
  if ( v44 != v8 )
  {
    for ( i = *(__int64 **)(a2 + 64); v44 != i; ++i )
    {
      v10 = *i;
      v45 = *(_QWORD *)(*i + 8) & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v45 )
        continue;
      v42 = *(_QWORD *)(*i + 8);
      v11 = (__int64 *)sub_2E09D00((__int64 *)a2, v42);
      if ( v11 == (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
        || (*(_DWORD *)((*v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v11 >> 1) & 3) > (*(_DWORD *)(v45 + 24)
                                                                                              | (unsigned int)(v42 >> 1)
                                                                                              & 3)
        || (v12 = v11[2]) == 0 )
      {
        sub_2EEFF60(a1, "Value not live at VNInfo def and not marked unused", *(__int64 **)(a1 + 32));
        goto LABEL_4;
      }
      if ( v10 != v12 )
      {
        sub_2EEFF60(a1, "Live segment at def has different VNInfo", *(__int64 **)(a1 + 32));
        goto LABEL_4;
      }
      v13 = *(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL);
      v14 = *(_QWORD *)((*(_QWORD *)(v10 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 16);
      if ( v14 )
      {
        v46 = *(_QWORD *)(v14 + 24);
        if ( v46 )
          goto LABEL_14;
      }
      else
      {
        v35 = *(unsigned int *)(v13 + 304);
        v36 = *(_QWORD **)(v13 + 296);
        v49[0] = *(_QWORD *)(v10 + 8);
        v46 = *(sub_2EEE710(v36, (__int64)&v36[2 * v35], v49) - 1);
        if ( v46 )
        {
LABEL_14:
          v15 = *(_QWORD *)(v10 + 8);
          v40 = (v15 >> 1) & 3;
          if ( ((v15 >> 1) & 3) != 0 )
          {
            v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
            v17 = v46;
            v18 = "No instruction at VNInfo def index";
            v19 = *(_QWORD *)(v16 + 16);
            if ( v19 )
            {
              v20 = *(_QWORD *)(v16 + 16);
              if ( (*(_BYTE *)(v19 + 44) & 4) != 0 )
              {
                do
                  v20 = *(_QWORD *)v20 & 0xFFFFFFFFFFFFFFF8LL;
                while ( (*(_BYTE *)(v20 + 44) & 4) != 0 );
              }
              v21 = *(_QWORD *)(v19 + 24) + 48LL;
              do
              {
                v22 = *(_QWORD *)(v20 + 32);
                v23 = v22 + 40LL * (*(_DWORD *)(v20 + 40) & 0xFFFFFF);
                if ( v22 != v23 )
                  goto LABEL_22;
                v20 = *(_QWORD *)(v20 + 8);
              }
              while ( v21 != v20 && (*(_BYTE *)(v20 + 44) & 4) != 0 );
              v20 = *(_QWORD *)(v19 + 24) + 48LL;
              if ( v23 == v22 )
              {
                v43 = 0;
                goto LABEL_58;
              }
LABEL_22:
              v43 = 0;
              v39 = 0;
LABEL_23:
              if ( !*(_BYTE *)v22 && (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
              {
                v24 = *(unsigned int *)(v22 + 8);
                if ( a3 >= 0 )
                {
                  if ( (unsigned int)(v24 - 1) <= 0x3FFFFFFE )
                  {
                    v25 = *(_QWORD *)(a1 + 56);
                    v26 = *(_DWORD *)(*(_QWORD *)(v25 + 8) + 24 * v24 + 16);
                    v27 = v26 & 0xFFF;
                    v28 = (__int16 *)(*(_QWORD *)(v25 + 56) + 2LL * (v26 >> 12));
                    do
                    {
                      if ( !v28 )
                        break;
                      if ( a3 == v27 )
                        goto LABEL_40;
                      v29 = *v28++;
                      v27 += v29;
                    }
                    while ( (_WORD)v29 );
                  }
                  goto LABEL_31;
                }
                if ( a3 != (_DWORD)v24 )
                  goto LABEL_31;
LABEL_40:
                if ( v38 )
                {
                  v37 = (_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 272LL) + 16LL * ((*(_DWORD *)v22 >> 8) & 0xFFF));
                  if ( !(v37[1] & a5 | *v37 & a4) )
                    goto LABEL_31;
                }
                v32 = v43;
                v39 = 1;
                if ( (*(_BYTE *)(v22 + 4) & 4) != 0 )
                  v32 = 1;
                v30 = v22 + 40;
                v43 = v32;
                v31 = v23;
                if ( v30 == v23 )
                {
LABEL_35:
                  while ( 1 )
                  {
                    v20 = *(_QWORD *)(v20 + 8);
                    if ( v21 == v20 || (*(_BYTE *)(v20 + 44) & 4) == 0 )
                      break;
                    v23 = *(_QWORD *)(v20 + 32);
                    v31 = v23 + 40LL * (*(_DWORD *)(v20 + 40) & 0xFFFFFF);
                    if ( v23 != v31 )
                      goto LABEL_38;
                  }
                  if ( v23 == v31 )
                  {
                    if ( v39 )
                    {
                      if ( v43 )
                        goto LABEL_54;
LABEL_59:
                      if ( v40 != 2 )
                      {
                        sub_2EF03A0(a1, "Non-PHI, non-early clobber def must be at a register slot", v46);
                        goto LABEL_4;
                      }
                      continue;
                    }
LABEL_58:
                    sub_2EF06E0(a1, "Defining instruction does not modify register", v19);
                    sub_2EEFB40(a1, a2, a3, a4, a5);
                    sub_2EEF900(*(_QWORD *)(a1 + 16), (unsigned int *)v10);
                    v40 = (*(__int64 *)(v10 + 8) >> 1) & 3;
                    if ( !v43 )
                      goto LABEL_59;
LABEL_54:
                    if ( v40 != 1 )
                    {
                      sub_2EF03A0(a1, "Early clobber def must be at an early-clobber slot", v46);
                      goto LABEL_4;
                    }
                    continue;
                  }
                  v20 = *(_QWORD *)(v19 + 24) + 48LL;
                }
                else
                {
LABEL_44:
                  v23 = v30;
                }
LABEL_38:
                v22 = v23;
                v23 = v31;
                goto LABEL_23;
              }
LABEL_31:
              v30 = v22 + 40;
              v31 = v23;
              if ( v30 == v23 )
                goto LABEL_35;
              goto LABEL_44;
            }
          }
          else
          {
            if ( v15 == *(_QWORD *)(*(_QWORD *)(v13 + 152) + 16LL * *(unsigned int *)(v46 + 24)) )
              continue;
            v17 = v46;
            v18 = "PHIDef VNInfo is not defined at MBB start";
          }
          sub_2EF03A0(a1, v18, v17);
          goto LABEL_4;
        }
      }
      sub_2EEFF60(a1, "Invalid VNInfo definition index", *(__int64 **)(a1 + 32));
LABEL_4:
      sub_2EEFB40(a1, a2, a3, a4, a5);
      sub_2EEF900(*(_QWORD *)(a1 + 16), (unsigned int *)v10);
    }
  }
  if ( *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8) != *(_QWORD *)a2 )
  {
    v47 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
    v33 = *(_QWORD *)a2;
    do
    {
      v34 = (_QWORD *)v33;
      v33 += 24;
      sub_2EF1130(a1, a2, v34, a3, a4, a5);
    }
    while ( v33 != v47 );
  }
}

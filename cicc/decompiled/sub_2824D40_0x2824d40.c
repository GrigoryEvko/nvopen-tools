// Function: sub_2824D40
// Address: 0x2824d40
//
__int64 __fastcall sub_2824D40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rdi
  unsigned int v7; // r15d
  bool v8; // al
  unsigned int v9; // r15d
  __int64 v11; // rax
  unsigned __int8 *v12; // r13
  unsigned int v13; // r11d
  __int64 *v14; // rax
  __int64 v15; // r14
  __int64 *v16; // rax
  __int64 *v17; // r9
  unsigned __int8 *v18; // r10
  __int64 v19; // rax
  __int64 v20; // rdx
  bool v21; // al
  __int64 v22; // r9
  __int64 v23; // r10
  char v24; // dl
  _QWORD *v25; // rax
  __int64 v26; // rax
  char v27; // si
  _QWORD *v28; // rdx
  __int64 v29; // rdx
  unsigned __int16 v30; // r15
  __int64 v31; // rdx
  __int64 v32; // rcx
  _QWORD *v33; // [rsp-8h] [rbp-C8h]
  __int64 v34; // [rsp+10h] [rbp-B0h]
  __int64 v35; // [rsp+10h] [rbp-B0h]
  __int64 v36; // [rsp+18h] [rbp-A8h]
  __int64 v37; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *v38; // [rsp+20h] [rbp-A0h]
  unsigned int v39; // [rsp+20h] [rbp-A0h]
  __int64 *v40; // [rsp+28h] [rbp-98h]
  unsigned __int16 v41; // [rsp+28h] [rbp-98h]
  bool v42; // [rsp+28h] [rbp-98h]
  __int64 v43; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v44; // [rsp+40h] [rbp-80h]
  unsigned __int64 v45; // [rsp+40h] [rbp-80h]
  __int64 v46; // [rsp+48h] [rbp-78h] BYREF
  _QWORD *v47; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-68h]
  _QWORD *v49; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v50; // [rsp+68h] [rbp-58h]
  __int64 v51[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v52[8]; // [rsp+80h] [rbp-40h] BYREF

  v46 = a2;
  v5 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v6 = *(_QWORD *)(a2 + 32 * (3 - v5));
  v7 = *(_DWORD *)(v6 + 32);
  if ( v7 <= 0x40 )
    v8 = *(_QWORD *)(v6 + 24) == 0;
  else
    v8 = v7 == (unsigned int)sub_C444A0(v6 + 24);
  if ( v8 && **(_BYTE **)(a2 + 32 * (2 - v5)) == 17 )
  {
    if ( *(_BYTE *)(a1 + 266) )
      goto LABEL_19;
    v11 = *(_QWORD *)(a2 - 32);
    if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a2 + 80) )
      BUG();
    if ( *(_DWORD *)(v11 + 36) == 240 )
    {
LABEL_19:
      if ( !unk_4FFF828 )
      {
        v12 = sub_BD3990(*(unsigned __int8 **)(a2 - 32 * v5), a2);
        v44 = sub_BD3990(*(unsigned __int8 **)(v46 + 32 * (1LL - (*(_DWORD *)(v46 + 4) & 0x7FFFFFF))), a2);
        LOBYTE(v13) = v44 == 0 || v12 == 0;
        v9 = v13;
        if ( !(_BYTE)v13 )
        {
          v14 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v12);
          v15 = (__int64)v14;
          if ( *((_WORD *)v14 + 12) == 8 && *(_QWORD *)a1 == v14[6] && v14[5] == 2 )
          {
            v16 = sub_DD8400(*(_QWORD *)(a1 + 32), (__int64)v44);
            v17 = v16;
            if ( *((_WORD *)v16 + 12) == 8 && *(_QWORD *)a1 == v16[6] && v16[5] == 2 )
            {
              v18 = v44;
              v19 = *(_QWORD *)(v46 + 32 * (2LL - (*(_DWORD *)(v46 + 4) & 0x7FFFFFF)));
              v45 = *(_DWORD *)(v19 + 32) <= 0x40u ? *(_QWORD *)(v19 + 24) : **(_QWORD **)(v19 + 24);
              if ( !HIDWORD(v45) )
              {
                v20 = *(_QWORD *)(*(_QWORD *)(v15 + 32) + 8LL);
                if ( !*(_WORD *)(v20 + 24) && !*(_WORD *)(*(_QWORD *)(v17[4] + 8) + 24LL) )
                {
                  v43 = *(_QWORD *)(v17[4] + 8);
                  v38 = v18;
                  v40 = v17;
                  sub_9865C0((__int64)&v47, *(_QWORD *)(v20 + 32) + 24LL);
                  sub_9865C0((__int64)&v49, *(_QWORD *)(v43 + 32) + 24LL);
                  if ( v48 > 0x40 )
                    goto LABEL_33;
                  v36 = (__int64)v38;
                  v39 = v48;
                  if ( v50 > 0x40 )
                    goto LABEL_33;
                  v21 = sub_D94970((__int64)&v47, (_QWORD *)v45);
                  v22 = (__int64)v40;
                  v23 = v36;
                  if ( v21 )
                  {
                    v24 = v39;
                    v25 = v47;
                    if ( !v39 )
                    {
                      v27 = v50;
                      v28 = v49;
                      if ( !v50 )
                        goto LABEL_36;
                      v26 = 0;
                      goto LABEL_38;
                    }
                    goto LABEL_29;
                  }
                  v35 = (__int64)v40;
                  sub_9865C0((__int64)v51, (__int64)&v47);
                  sub_AADAA0((__int64)v52, (__int64)v51, v31, v32, (__int64)v51);
                  v42 = sub_D94970((__int64)v52, (_QWORD *)v45);
                  sub_969240(v52);
                  sub_969240(v51);
                  v22 = v35;
                  v23 = v36;
                  if ( !v42 )
                  {
                    v9 = 0;
                    sub_2822A00(*(__int64 **)(a1 + 64), &v46);
                    goto LABEL_33;
                  }
                  v24 = v48;
                  v25 = v47;
                  if ( v48 <= 0x40 )
                  {
                    if ( v48 )
                    {
LABEL_29:
                      v26 = (__int64)((_QWORD)v25 << (64 - v24)) >> (64 - v24);
                      goto LABEL_30;
                    }
                    v26 = 0;
                  }
                  else
                  {
                    v26 = *v47;
                  }
LABEL_30:
                  v27 = v50;
                  v28 = v49;
                  if ( v50 > 0x40 )
                  {
                    v29 = *v49;
                    goto LABEL_32;
                  }
                  if ( !v50 )
                  {
                    v29 = 0;
LABEL_32:
                    if ( v26 != v29 )
                    {
LABEL_33:
                      sub_969240((__int64 *)&v49);
                      sub_969240((__int64 *)&v47);
                      return v9;
                    }
LABEL_36:
                    v34 = v23;
                    v37 = v22;
                    v41 = sub_A74840((_QWORD *)(v46 + 72), 1);
                    v30 = sub_A74840((_QWORD *)(v46 + 72), 0);
                    v33 = sub_DA2C50(*(_QWORD *)(a1 + 32), *((_QWORD *)v12 + 1), v45, 0);
                    v9 = sub_2822E30(a1, (__int64)v12, v34, (__int64)v33, v30, v41, v46, v46, v15, v37, a3);
                    goto LABEL_33;
                  }
LABEL_38:
                  v29 = (__int64)((_QWORD)v28 << (64 - v27)) >> (64 - v27);
                  goto LABEL_32;
                }
              }
            }
          }
        }
      }
    }
  }
  return 0;
}

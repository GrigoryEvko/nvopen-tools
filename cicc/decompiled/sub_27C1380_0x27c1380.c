// Function: sub_27C1380
// Address: 0x27c1380
//
__int64 __fastcall sub_27C1380(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 i; // r15
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int8 *v17; // r12
  __int64 *v18; // r14
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  unsigned __int64 v21; // r12
  unsigned __int8 *v22; // rdi
  unsigned __int8 *v23; // rsi
  __int64 v24; // r10
  char v25; // al
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r9
  unsigned __int64 v30; // rax
  int v31; // edx
  __int64 v32; // rsi
  __int64 v33; // rax
  __int64 v34; // r14
  char v36; // [rsp+8h] [rbp-F8h]
  bool v37; // [rsp+8h] [rbp-F8h]
  __int64 v38; // [rsp+10h] [rbp-F0h]
  __int64 v40; // [rsp+20h] [rbp-E0h]
  __int64 v41; // [rsp+28h] [rbp-D8h]
  __int64 v42; // [rsp+38h] [rbp-C8h]
  __int64 v45; // [rsp+58h] [rbp-A8h]
  __int64 v46; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v47; // [rsp+68h] [rbp-98h]
  unsigned __int64 v48; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v49; // [rsp+78h] [rbp-88h]
  __int64 v50; // [rsp+80h] [rbp-80h]
  int v51; // [rsp+88h] [rbp-78h]
  char v52; // [rsp+8Ch] [rbp-74h]
  __int64 v53; // [rsp+90h] [rbp-70h] BYREF

  v6 = sub_D95540(a3);
  v47 = sub_D97050(a4, v6);
  v40 = a2 + 48;
  v7 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( a2 + 48 == v7 )
    goto LABEL_56;
  if ( !v7 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v7 - 24) - 30 > 0xA )
LABEL_56:
    BUG();
  v38 = *(_QWORD *)(v7 - 120);
  v42 = sub_D47930(a1);
  v41 = 0;
  v45 = sub_AA4E30(**(_QWORD **)(a1 + 32));
  v46 = 0;
  for ( i = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL); ; i = *(_QWORD *)(i + 8) )
  {
    if ( !i )
      BUG();
    if ( *(_BYTE *)(i - 24) != 84 )
      break;
    if ( sub_D97040(a4, *(_QWORD *)(i - 16)) )
    {
      v9 = i - 24;
      v10 = sub_DD8400(a4, i - 24);
      if ( *((_WORD *)v10 + 12) == 8 && a1 == v10[6] && v10[5] == 2 )
      {
        v11 = *(_QWORD *)(v10[4] + 8);
        if ( !*(_WORD *)(v11 + 24) && sub_D96900(v11) )
        {
          v12 = sub_D47930(a1);
          v13 = *(_QWORD *)(i - 32);
          v14 = v12;
          if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
          {
            v15 = 0;
            while ( v14 != *(_QWORD *)(v13 + 32LL * *(unsigned int *)(i + 48) + 8 * v15) )
            {
              if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) == (_DWORD)++v15 )
                goto LABEL_42;
            }
            v16 = 32 * v15;
          }
          else
          {
LABEL_42:
            v16 = 0x1FFFFFFFE0LL;
          }
          v17 = *(unsigned __int8 **)(v13 + v16);
          if ( v9 == sub_27C0260(v17, a1) && *((_WORD *)sub_DD8400(a4, (__int64)v17) + 12) == 8 )
          {
            v18 = sub_DD8400(a4, i - 24);
            v19 = sub_D95540(*(_QWORD *)v18[4]);
            v20 = sub_D97050(a4, v19);
            v21 = v20;
            if ( v47 <= v20 )
            {
              v48 = v20;
              v22 = *(unsigned __int8 **)(v45 + 32);
              v23 = &v22[*(_QWORD *)(v45 + 40)];
              if ( v23 != sub_27BFE20(v22, (__int64)v23, (__int64 *)&v48) )
              {
                v51 = 0;
                v49 = &v53;
                v50 = 0x100000008LL;
                v52 = 1;
                v53 = i - 24;
                v48 = 1;
                v25 = sub_27C1230(i - 24, v24, 0);
                if ( !v52 )
                {
                  v36 = v25;
                  _libc_free((unsigned __int64)v49);
                  v25 = v36;
                }
                if ( v25 )
                  goto LABEL_59;
                if ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != 0 )
                {
                  v28 = 0;
                  do
                  {
                    if ( v42 == *(_QWORD *)(*(_QWORD *)(i - 32) + 32LL * *(unsigned int *)(i + 48) + 8 * v28) )
                      break;
                    ++v28;
                  }
                  while ( (*(_DWORD *)(i - 20) & 0x7FFFFFF) != (_DWORD)v28 );
                }
                if ( sub_27C10F0(i - 24, a2) || sub_27C10F0(*(_QWORD *)(v29 + v27), v26) )
                {
LABEL_59:
                  if ( *(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) == 12 )
                    goto LABEL_38;
                  v30 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
                  if ( v40 == v30 )
                  {
                    v32 = 0;
                  }
                  else
                  {
                    if ( !v30 )
                      BUG();
                    v31 = *(unsigned __int8 *)(v30 - 24);
                    v32 = 0;
                    v33 = v30 - 24;
                    if ( (unsigned int)(v31 - 30) < 0xB )
                      v32 = v33;
                  }
                  if ( (unsigned __int8)sub_98EF90(i - 24, v32, a5, v26, v27) )
                  {
LABEL_38:
                    v34 = *(_QWORD *)v18[4];
                    if ( v46 && !(unsigned __int8)sub_F6EBA0(v46, v42, v38) )
                    {
                      if ( !(unsigned __int8)sub_F6EBA0(i - 24, v42, v38) )
                      {
                        v37 = sub_D968A0(v41);
                        if ( v37 == sub_D968A0(v34) )
                        {
                          if ( sub_D97050(a4, *(_QWORD *)(v46 + 8)) >= v21 )
                          {
                            v34 = v41;
                            v9 = v46;
                          }
                          v41 = v34;
                          v46 = v9;
                        }
                        else
                        {
                          if ( sub_D968A0(v41) )
                          {
                            v9 = v46;
                            v34 = v41;
                          }
                          v46 = v9;
                          v41 = v34;
                        }
                      }
                    }
                    else
                    {
                      v41 = v34;
                      v46 = i - 24;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return v46;
}

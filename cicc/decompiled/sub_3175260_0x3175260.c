// Function: sub_3175260
// Address: 0x3175260
//
unsigned __int64 __fastcall sub_3175260(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 *v5; // rax
  __int64 v6; // r9
  __int64 *v7; // r15
  __int64 v8; // r12
  __int64 *v9; // rbx
  unsigned __int64 v10; // r14
  __int64 v11; // r9
  __int64 **v12; // r11
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // r8
  unsigned __int64 v16; // rdx
  __int64 v17; // r13
  unsigned __int8 **v18; // rdi
  int v19; // ecx
  unsigned __int8 **v20; // r9
  unsigned __int64 v21; // rcx
  signed __int64 v22; // r12
  int v23; // edx
  int v24; // r13d
  signed __int64 v25; // rax
  unsigned __int64 v26; // rax
  bool v27; // cc
  __int64 v28; // [rsp+8h] [rbp-A8h]
  __int64 v29; // [rsp+10h] [rbp-A0h]
  __int64 **v30; // [rsp+18h] [rbp-98h]
  __int64 v31; // [rsp+20h] [rbp-90h]
  __int64 *v32; // [rsp+30h] [rbp-80h]
  __int64 v34; // [rsp+40h] [rbp-70h]
  unsigned __int64 v35; // [rsp+48h] [rbp-68h]
  unsigned __int8 **v36; // [rsp+50h] [rbp-60h] BYREF
  __int64 v37; // [rsp+58h] [rbp-58h]
  _BYTE v38[80]; // [rsp+60h] [rbp-50h] BYREF

  v3 = *(_QWORD *)(a1 + 32);
  if ( !*(_QWORD *)(a1 + 16) )
    sub_4263D6(a1, v3, a3);
  v32 = (__int64 *)(*(__int64 (__fastcall **)(__int64, __int64))(a1 + 24))(a1, v3);
  if ( *(_DWORD *)(a1 + 80) )
  {
    v5 = *(__int64 **)(a1 + 72);
    v6 = 2LL * *(unsigned int *)(a1 + 88);
    v7 = &v5[v6];
    if ( v5 != &v5[v6] )
    {
      while ( 1 )
      {
        v8 = *v5;
        v9 = v5;
        if ( *v5 != -4096 && v8 != -8192 )
          break;
        v5 += 2;
        if ( v7 == v5 )
          return 0;
      }
      if ( v5 != v7 )
      {
        v35 = 0;
        while ( 1 )
        {
          if ( *(_BYTE *)v8 <= 0x1Cu )
            goto LABEL_27;
          v10 = sub_FDD860(v32, *(_QWORD *)(v8 + 40));
          v34 = v10 / sub_FDC4B0((__int64)v32);
          v12 = *(__int64 ***)(a1 + 48);
          if ( (*(_BYTE *)(v8 + 7) & 0x40) != 0 )
          {
            v13 = *(_QWORD *)(v8 - 8);
            v14 = v13 + 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
          }
          else
          {
            v14 = v8;
            v13 = v8 - 32LL * (*(_DWORD *)(v8 + 4) & 0x7FFFFFF);
          }
          v15 = v14 - v13;
          v36 = (unsigned __int8 **)v38;
          v16 = v15 >> 5;
          v37 = 0x400000000LL;
          v17 = v15 >> 5;
          if ( (unsigned __int64)v15 > 0x80 )
          {
            v28 = v15;
            v29 = v13;
            v30 = v12;
            v31 = v15 >> 5;
            sub_C8D5F0((__int64)&v36, v38, v16, 8u, v15, v11);
            v20 = v36;
            v19 = v37;
            LODWORD(v16) = v31;
            v12 = v30;
            v13 = v29;
            v15 = v28;
            v18 = &v36[(unsigned int)v37];
          }
          else
          {
            v18 = (unsigned __int8 **)v38;
            v19 = 0;
            v20 = (unsigned __int8 **)v38;
          }
          if ( v15 > 0 )
          {
            v21 = 0;
            do
            {
              v18[v21 / 8] = *(unsigned __int8 **)(v13 + 4 * v21);
              v21 += 8LL;
              --v17;
            }
            while ( v17 );
            v20 = v36;
            v19 = v37;
          }
          LODWORD(v37) = v19 + v16;
          v22 = sub_DFCEF0(v12, (unsigned __int8 *)v8, v20, (unsigned int)(v19 + v16), 1);
          v24 = v23;
          if ( v36 != (unsigned __int8 **)v38 )
            _libc_free((unsigned __int64)v36);
          if ( v24 != 1 )
          {
            v25 = v22 * v34;
            if ( is_mul_ok(v22, v34) )
              goto LABEL_25;
            if ( v34 <= 0 )
            {
              if ( v22 >= 0 || v34 >= 0 )
              {
LABEL_44:
                v26 = v35 + 0x8000000000000000LL;
                if ( !__OFADD__(v35, 0x8000000000000000LL) )
                  goto LABEL_41;
LABEL_45:
                v35 = 0x8000000000000000LL;
                goto LABEL_27;
              }
            }
            else if ( v22 <= 0 )
            {
              goto LABEL_44;
            }
            v26 = v35 + 0x7FFFFFFFFFFFFFFFLL;
            if ( !__OFADD__(v35, 0x7FFFFFFFFFFFFFFFLL) )
              goto LABEL_41;
LABEL_46:
            v35 = 0x7FFFFFFFFFFFFFFFLL;
            goto LABEL_27;
          }
          v25 = v22 * v34;
          if ( is_mul_ok(v22, v34) )
          {
LABEL_25:
            if ( !__OFADD__(v25, v35) )
            {
              v35 += v25;
              goto LABEL_27;
            }
            v27 = v25 <= 0;
            v26 = 0x8000000000000000LL;
            if ( !v27 )
              v26 = 0x7FFFFFFFFFFFFFFFLL;
LABEL_41:
            v35 = v26;
            goto LABEL_27;
          }
          if ( v34 <= 0 )
          {
            if ( v22 >= 0 || v34 >= 0 )
            {
LABEL_56:
              if ( __OFADD__(v35, 0x8000000000000000LL) )
                goto LABEL_45;
              v35 += 0x8000000000000000LL;
              goto LABEL_27;
            }
          }
          else if ( v22 <= 0 )
          {
            goto LABEL_56;
          }
          if ( __OFADD__(v35, 0x7FFFFFFFFFFFFFFFLL) )
            goto LABEL_46;
          v35 += 0x7FFFFFFFFFFFFFFFLL;
LABEL_27:
          v9 += 2;
          if ( v9 != v7 )
          {
            while ( 1 )
            {
              v8 = *v9;
              if ( *v9 != -8192 && v8 != -4096 )
                break;
              v9 += 2;
              if ( v7 == v9 )
                return v35;
            }
            if ( v7 != v9 )
              continue;
          }
          return v35;
        }
      }
    }
  }
  return 0;
}

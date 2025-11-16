// Function: sub_3543BE0
// Address: 0x3543be0
//
_BOOL8 __fastcall sub_3543BE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rax
  __int64 *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rax
  __int64 (*v9)(void); // rdx
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  __int64 v12; // r13
  int v13; // eax
  int v14; // ecx
  __int64 v15; // rsi
  __int64 v16; // rax
  _DWORD *v18; // rax
  __int64 (*v19)(); // rax
  __int64 v20; // rbx
  _BYTE *v21; // r12
  _BYTE *v22; // r14
  _BYTE *v23; // rbx
  int v24; // r15d
  _BYTE *v25; // r15
  __int64 v26; // [rsp+0h] [rbp-B0h]
  __int64 v27; // [rsp+10h] [rbp-A0h]
  __int64 v28; // [rsp+18h] [rbp-98h]
  __int64 *v29; // [rsp+20h] [rbp-90h]
  __int64 v30; // [rsp+28h] [rbp-88h]
  int v31; // [rsp+34h] [rbp-7Ch]
  __int64 v32; // [rsp+38h] [rbp-78h]
  bool v34; // [rsp+4Bh] [rbp-65h]
  int v35; // [rsp+4Ch] [rbp-64h]
  char v36; // [rsp+5Eh] [rbp-52h] BYREF
  char v37; // [rsp+5Fh] [rbp-51h] BYREF
  __int64 v38; // [rsp+60h] [rbp-50h] BYREF
  _BYTE v39[8]; // [rsp+68h] [rbp-48h] BYREF
  __int64 v40; // [rsp+70h] [rbp-40h] BYREF
  _BYTE v41[56]; // [rsp+78h] [rbp-38h] BYREF

  v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL));
  if ( (unsigned __int8)sub_2FE0930(*(__int64 **)(a1 + 16), a2, &v38, (__int64)v39, (__int64)&v36, v3) )
  {
    if ( !v36 && !*(_BYTE *)v38 )
    {
      v31 = *(_DWORD *)(v38 + 8);
      if ( v31 < 0 )
      {
        v29 = 0;
        v4 = *(_QWORD *)(*(_QWORD *)(v38 + 16) + 24LL);
        v5 = *(_QWORD *)(v4 + 32);
        v6 = *(__int64 **)(v5 + 16);
        v7 = *(_QWORD *)(v5 + 32);
        v8 = *v6;
        v32 = v7;
        v9 = *(__int64 (**)(void))(*v6 + 128);
        if ( v9 != sub_2DAC790 )
        {
          v29 = (__int64 *)v9();
          v8 = **(_QWORD **)(*(_QWORD *)(v4 + 32) + 16LL);
        }
        v10 = 0;
        v30 = 0;
        v27 = (*(__int64 (**)(void))(v8 + 200))();
        v35 = v31;
        do
        {
          v11 = sub_2EBEE10(v32, v35);
          v12 = v11;
          if ( v4 != *(_QWORD *)(v11 + 24) )
            break;
          v13 = *(unsigned __int16 *)(v11 + 68);
          if ( (_WORD)v13 == 20 )
          {
            v18 = *(_DWORD **)(v12 + 32);
            if ( (*v18 & 0xFFF00) != 0 || (v18[10] & 0xFFF00) != 0 )
              break;
            v35 = v18[12];
          }
          else
          {
            v34 = v13 == 68 || v13 == 0;
            if ( v34 )
            {
              if ( v30 )
                break;
              v14 = *(_DWORD *)(v12 + 40) & 0xFFFFFF;
              if ( v14 == 1 )
                break;
              v15 = *(_QWORD *)(v12 + 32);
              v16 = 1;
              while ( v4 != *(_QWORD *)(v15 + 40LL * (unsigned int)(v16 + 1) + 24) )
              {
                v16 = (unsigned int)(v16 + 2);
                if ( v14 == (_DWORD)v16 )
                  return 0;
              }
              v30 = v12;
              v35 = *(_DWORD *)(v15 + 40 * v16 + 8);
            }
            else
            {
              v19 = *(__int64 (**)())(*v29 + 864);
              if ( v19 == sub_2FDC6E0
                || !((unsigned __int8 (__fastcall *)(__int64 *, __int64, __int64))v19)(v29, v12, a3) )
              {
                break;
              }
              if ( v10 )
                return v34;
              if ( !(unsigned __int8)sub_2FE0930(v29, v12, &v40, (__int64)v41, (__int64)&v37, v27) )
              {
                v20 = *(_QWORD *)(v12 + 32);
                v26 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v12 + 24) + 32LL) + 32LL);
                v21 = (_BYTE *)(v20 + 40LL * (*(_DWORD *)(v12 + 40) & 0xFFFFFF));
                v22 = (_BYTE *)(v20 + 40LL * (unsigned int)sub_2E88FE0(v12));
                if ( v21 != v22 )
                {
                  while ( 1 )
                  {
                    v23 = v22;
                    if ( (unsigned __int8)sub_2E2FA70(v22) )
                      break;
                    v22 += 40;
                    if ( v21 == v22 )
                      return v34;
                  }
                  if ( v21 != v22 )
                  {
                    v35 = 0;
                    v28 = v4;
                    while ( 1 )
                    {
                      v24 = *((_DWORD *)v23 + 2);
                      if ( v24 >= 0 )
                        break;
                      if ( *(_QWORD *)(sub_2EBEE10(v26, v24) + 24) == *(_QWORD *)(v12 + 24) )
                      {
                        if ( v35 )
                          return v34;
                        v35 = v24;
                      }
                      if ( v23 + 40 != v21 )
                      {
                        v25 = v23 + 40;
                        while ( 1 )
                        {
                          v23 = v25;
                          if ( (unsigned __int8)sub_2E2FA70(v25) )
                            break;
                          v25 += 40;
                          if ( v21 == v25 )
                            goto LABEL_43;
                        }
                        if ( v21 != v25 )
                          continue;
                      }
LABEL_43:
                      v4 = v28;
                      if ( v35 )
                        goto LABEL_29;
                      return v34;
                    }
                  }
                }
                return v34;
              }
              v35 = *(_DWORD *)(v40 + 8);
LABEL_29:
              v10 = v12;
            }
          }
          if ( v31 == v35 )
            return v10 != 0 && v30 != 0;
        }
        while ( v35 < 0 );
      }
    }
  }
  return 0;
}

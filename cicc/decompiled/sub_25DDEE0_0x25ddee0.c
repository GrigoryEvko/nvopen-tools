// Function: sub_25DDEE0
// Address: 0x25ddee0
//
__int64 __fastcall sub_25DDEE0(__int64 a1, unsigned __int64 a2)
{
  unsigned int v3; // r14d
  __int64 v5; // rcx
  unsigned __int8 *v6; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r9
  unsigned int v10; // eax
  __int64 v11; // rax
  __int64 v12; // rax
  int v14; // edx
  unsigned int v15; // r8d
  unsigned int v16; // r9d
  __int64 v17; // rcx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  __int64 *v21; // r11
  __int64 v22; // rcx
  unsigned int v23; // edx
  unsigned int v24; // esi
  unsigned __int64 v25; // r8
  unsigned __int8 *v26; // rax
  _BYTE *v27; // r8
  unsigned __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r14
  __int64 v32; // rax
  int v33; // r14d
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r14
  unsigned __int8 *v40; // rdi
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // r11
  __int64 v44; // rsi
  __int64 v45; // rdx
  int v46; // eax
  unsigned int v47; // eax
  __int64 v48; // [rsp-8h] [rbp-D8h]
  _BYTE *v49; // [rsp+0h] [rbp-D0h]
  unsigned __int8 v50; // [rsp+8h] [rbp-C8h]
  __int64 v51; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v52; // [rsp+8h] [rbp-C8h]
  bool v53; // [rsp+10h] [rbp-C0h]
  __int64 v54; // [rsp+10h] [rbp-C0h]
  unsigned __int8 v55; // [rsp+10h] [rbp-C0h]
  __int64 *v56; // [rsp+10h] [rbp-C0h]
  __int64 *v57; // [rsp+10h] [rbp-C0h]
  bool v58; // [rsp+18h] [rbp-B8h]
  __int64 *v59; // [rsp+18h] [rbp-B8h]
  __int64 *v60; // [rsp+18h] [rbp-B8h]
  __int64 *v61; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v62; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v63; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v64; // [rsp+30h] [rbp-A0h]
  unsigned int v65; // [rsp+38h] [rbp-98h]
  char v66; // [rsp+40h] [rbp-90h]
  __int64 *v67; // [rsp+50h] [rbp-80h] BYREF
  __int64 v68; // [rsp+58h] [rbp-78h]
  _BYTE v69[112]; // [rsp+60h] [rbp-70h] BYREF

  v3 = 0;
  v5 = *(_QWORD *)(a1 + 16);
  if ( v5 )
  {
    while ( 1 )
    {
      v6 = *(unsigned __int8 **)(v5 + 24);
      v7 = *(_QWORD *)(v5 + 8);
      v8 = sub_B43CB0((__int64)v6);
      if ( sub_B2F070(v8, 0) )
        return 0;
      v10 = *v6;
      if ( (_BYTE)v10 == 61 )
      {
        if ( *((_QWORD *)v6 - 4) )
        {
          v11 = *((_QWORD *)v6 - 3);
          **((_QWORD **)v6 - 2) = v11;
          if ( v11 )
            *(_QWORD *)(v11 + 16) = *((_QWORD *)v6 - 2);
        }
        *((_QWORD *)v6 - 4) = a2;
        if ( a2 )
        {
          v12 = *(_QWORD *)(a2 + 16);
          *((_QWORD *)v6 - 3) = v12;
          if ( v12 )
            *(_QWORD *)(v12 + 16) = v6 - 24;
          *((_QWORD *)v6 - 2) = a2 + 16;
          *(_QWORD *)(a2 + 16) = v6 - 32;
        }
        v5 = v7;
        v3 = 1;
      }
      else if ( (_BYTE)v10 == 62 )
      {
        v5 = v7;
        LOBYTE(v10) = *((_QWORD *)v6 - 4) != 0 && a1 == *((_QWORD *)v6 - 4);
        if ( (_BYTE)v10 )
        {
          v3 = v10;
          sub_AC2B30((__int64)(v6 - 32), a2);
          v5 = v7;
        }
      }
      else
      {
        LOBYTE(v9) = (_BYTE)v10 == 85 || (_BYTE)v10 == 34;
        if ( !(_BYTE)v9 )
        {
          if ( (_BYTE)v10 == 79 )
          {
            v18 = sub_ADA8A0(a2, *((_QWORD *)v6 + 1), 0);
            v20 = sub_25DDEE0(v6, v18, v19);
            v5 = v7;
            v3 |= v20;
            if ( !*((_QWORD *)v6 + 2) )
            {
              v3 = 1;
              sub_B43D60(v6);
              v5 = v7;
            }
          }
          else
          {
            v5 = v7;
            if ( (_BYTE)v10 == 63 )
            {
              v21 = (__int64 *)v69;
              v22 = 0;
              v23 = 0;
              v67 = (__int64 *)v69;
              v68 = 0x800000000LL;
              v24 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
              v25 = v24 - 1;
              if ( v25 > 8 )
              {
                sub_C8D5F0((__int64)&v67, v69, v25, 8u, v25, v9);
                v22 = (unsigned int)v68;
                v21 = (__int64 *)v69;
                v24 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
                v23 = v68;
              }
              v26 = &v6[32 * (1LL - v24)];
              if ( v6 != v26 )
              {
                while ( 1 )
                {
                  v27 = *(_BYTE **)v26;
                  v22 = v23;
                  if ( **(_BYTE **)v26 > 0x15u )
                    break;
                  v28 = v23 + 1LL;
                  if ( v22 + 1 > (unsigned __int64)HIDWORD(v68) )
                  {
                    v49 = *(_BYTE **)v26;
                    v52 = v26;
                    v56 = v21;
                    sub_C8D5F0((__int64)&v67, v21, v28, 8u, (__int64)v27, v9);
                    v22 = (unsigned int)v68;
                    v27 = v49;
                    v26 = v52;
                    v21 = v56;
                  }
                  v26 += 32;
                  v67[v22] = (__int64)v27;
                  v23 = v68 + 1;
                  LODWORD(v68) = v68 + 1;
                  if ( v6 == v26 )
                  {
                    v22 = v23;
                    v24 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
                    goto LABEL_36;
                  }
                }
                v24 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
              }
LABEL_36:
              if ( v24 - 1 == v23 )
              {
                v41 = *((_QWORD *)v6 + 9);
                v59 = v21;
                v66 = 0;
                v42 = sub_AD9FD0(v41, (unsigned __int8 *)a2, v67, v22, 0, (__int64)&v62, 0);
                v43 = v59;
                v44 = v42;
                v45 = v48;
                if ( v66 )
                {
                  v66 = 0;
                  if ( v65 > 0x40 && v64 )
                  {
                    j_j___libc_free_0_0(v64);
                    v43 = v59;
                  }
                  if ( v63 > 0x40 && v62 )
                  {
                    v57 = v43;
                    j_j___libc_free_0_0(v62);
                    v43 = v57;
                  }
                }
                v60 = v43;
                v46 = sub_25DDEE0(v6, v44, v45);
                v21 = v60;
                v3 |= v46;
              }
              if ( !*((_QWORD *)v6 + 2) )
              {
                v61 = v21;
                v3 = 1;
                sub_B43D60(v6);
                v21 = v61;
              }
              if ( v67 != v21 )
                _libc_free((unsigned __int64)v67);
              v5 = v7;
            }
          }
          goto LABEL_12;
        }
        v5 = v7;
        if ( a1 == *((_QWORD *)v6 - 4) )
        {
          v53 = (_BYTE)v10 == 85 || (_BYTE)v10 == 34;
          sub_AC2B30((__int64)(v6 - 32), a2);
          v14 = *v6;
          v15 = 0;
          v16 = v53;
          if ( v14 == 40 )
          {
            v47 = sub_B491D0((__int64)v6);
            v15 = 0;
            v16 = v53;
            v17 = 32LL * v47;
          }
          else
          {
            v17 = 0;
            if ( v14 != 85 )
            {
              if ( v14 != 34 )
                BUG();
              v17 = 64;
            }
          }
          if ( (v6[7] & 0x80u) != 0 )
          {
            v50 = v16;
            v54 = v17;
            v29 = sub_BD2BC0((__int64)v6);
            v15 = 0;
            v17 = v54;
            v16 = v50;
            v31 = v29 + v30;
            if ( (v6[7] & 0x80u) == 0 )
            {
              if ( (unsigned int)(v31 >> 4) )
LABEL_72:
                BUG();
            }
            else
            {
              v32 = sub_BD2BC0((__int64)v6);
              v15 = 0;
              v17 = v54;
              v16 = v50;
              if ( (unsigned int)((v31 - v32) >> 4) )
              {
                if ( (v6[7] & 0x80u) == 0 )
                  goto LABEL_72;
                v33 = *(_DWORD *)(sub_BD2BC0((__int64)v6) + 8);
                if ( (v6[7] & 0x80u) == 0 )
                  BUG();
                v34 = sub_BD2BC0((__int64)v6);
                v16 = v50;
                v17 = v54;
                v15 = 0;
                v36 = 32LL * (unsigned int)(*(_DWORD *)(v34 + v35 - 4) - v33);
                goto LABEL_50;
              }
            }
          }
          v36 = 0;
LABEL_50:
          v37 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
          v38 = (32 * v37 - 32 - v17 - v36) >> 5;
          if ( !(_DWORD)v38 )
            goto LABEL_62;
          v38 = (unsigned int)v38;
          v39 = 0;
          while ( 1 )
          {
            v40 = &v6[32 * (v39 - v37)];
            if ( a1 == *(_QWORD *)v40 && *(_QWORD *)v40 != 0 )
            {
              v51 = v38;
              v55 = v16;
              v58 = a1 == *(_QWORD *)v40 && *(_QWORD *)v40 != 0;
              sub_AC2B30((__int64)v40, a2);
              v38 = v51;
              v16 = v55;
              v15 = v58;
            }
            if ( v38 == ++v39 )
              break;
            v37 = *((_DWORD *)v6 + 1) & 0x7FFFFFF;
          }
          if ( (_BYTE)v15 )
          {
            v5 = *(_QWORD *)(a1 + 16);
            v3 = v15;
          }
          else
          {
LABEL_62:
            v5 = v7;
            v3 = v16;
          }
        }
      }
LABEL_12:
      if ( !v5 )
        return v3;
    }
  }
  return 0;
}

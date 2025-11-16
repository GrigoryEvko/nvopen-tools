// Function: sub_2406C80
// Address: 0x2406c80
//
__int64 __fastcall sub_2406C80(
        __int64 *a1,
        char a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        unsigned int **a6,
        __int64 *a7)
{
  __int64 v7; // r13
  __int64 v9; // rdi
  __int64 *v10; // rax
  __int64 v11; // rax
  unsigned int *v12; // rdi
  unsigned __int8 *v13; // rbx
  __int64 (__fastcall *v14)(__int64, unsigned int, unsigned __int8 *, _BYTE *); // rax
  __int64 v15; // r14
  __int64 v16; // rsi
  __int64 v17; // rbx
  __int64 v18; // rax
  __int64 result; // rax
  _QWORD *v20; // rax
  __int64 v21; // rbx
  unsigned int *v22; // r14
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 *v27; // rax
  unsigned __int8 v28; // si
  int v29; // eax
  __int64 v30; // rdi
  int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // r9
  __int64 v34; // rsi
  __int64 *v35; // rdi
  char v36; // al
  unsigned int *v37; // rbx
  __int64 v38; // r13
  __int64 v39; // rdx
  unsigned int v40; // esi
  int v41; // edi
  __int64 v42; // r9
  int v43; // edi
  unsigned int v44; // eax
  __int64 v45; // r11
  int v46; // r10d
  int v47; // edx
  __int64 v48; // [rsp+0h] [rbp-A0h]
  __int64 *v49; // [rsp+0h] [rbp-A0h]
  __int64 v50; // [rsp+0h] [rbp-A0h]
  __int64 *v51; // [rsp+8h] [rbp-98h]
  __int64 v52; // [rsp+8h] [rbp-98h]
  __int64 *v53; // [rsp+8h] [rbp-98h]
  __int64 v54[4]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v55; // [rsp+30h] [rbp-70h]
  _BYTE v56[32]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v57; // [rsp+60h] [rbp-40h]

  v7 = a3;
  if ( !a2 )
  {
    if ( *(_BYTE *)a3 != 82 )
      goto LABEL_3;
    v25 = *(_QWORD *)(a3 + 16);
    v26 = v25;
    if ( v25 )
    {
      while ( 1 )
      {
        v27 = *(__int64 **)(v26 + 24);
        if ( a4 != v27 )
        {
          v28 = *(_BYTE *)v27;
          if ( *(_BYTE *)v27 <= 0x1Cu )
            goto LABEL_3;
          if ( v28 == 31 )
          {
            if ( (*((_DWORD *)v27 + 1) & 0x7FFFFFF) != 3 )
              goto LABEL_3;
          }
          else if ( v28 != 86 || v7 != *(v27 - 12) )
          {
LABEL_3:
            v9 = *a1;
            v55 = 257;
            v10 = (__int64 *)sub_B2BE50(v9);
            v11 = sub_ACD6D0(v10);
            v12 = a6[10];
            v13 = (unsigned __int8 *)v11;
            v14 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, _BYTE *))(*(_QWORD *)v12 + 16LL);
            if ( v14 == sub_9202E0 )
            {
              if ( *v13 > 0x15u || *(_BYTE *)v7 > 0x15u )
                goto LABEL_37;
              if ( (unsigned __int8)sub_AC47B0(30) )
                v15 = sub_AD5570(30, (__int64)v13, (unsigned __int8 *)v7, 0, 0);
              else
                v15 = sub_AABE40(0x1Eu, v13, (unsigned __int8 *)v7);
            }
            else
            {
              v15 = v14((__int64)v12, 30u, v13, (_BYTE *)v7);
            }
            if ( v15 )
            {
LABEL_9:
              v7 = v15;
              goto LABEL_11;
            }
LABEL_37:
            v57 = 257;
            v15 = sub_B504D0(30, (__int64)v13, v7, (__int64)v56, 0, 0);
            (*(void (__fastcall **)(unsigned int *, __int64, __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)a6[11] + 16LL))(
              a6[11],
              v15,
              v54,
              a6[7],
              a6[8]);
            v37 = *a6;
            v38 = (__int64)&(*a6)[4 * *((unsigned int *)a6 + 2)];
            if ( *a6 != (unsigned int *)v38 )
            {
              do
              {
                v39 = *((_QWORD *)v37 + 1);
                v40 = *v37;
                v37 += 4;
                sub_B99FD0(v15, v40, v39);
              }
              while ( (unsigned int *)v38 != v37 );
            }
            goto LABEL_9;
          }
        }
        v26 = *(_QWORD *)(v26 + 8);
        if ( !v26 )
        {
          while ( 1 )
          {
            v35 = *(__int64 **)(v25 + 24);
            if ( a4 == v35 )
              goto LABEL_33;
            v36 = *(_BYTE *)v35;
            if ( *(_BYTE *)v35 <= 0x1Cu )
LABEL_55:
              BUG();
            if ( v36 == 31 )
            {
              v50 = a5;
              v53 = a4;
              sub_B4CC70((__int64)v35);
              a4 = v53;
              a5 = v50;
              goto LABEL_33;
            }
            if ( v36 != 86 )
              goto LABEL_55;
            v54[0] = *(_QWORD *)(v25 + 24);
            v48 = a5;
            v51 = a4;
            sub_BD28A0(v35 - 8, v35 - 4);
            sub_B47280(v54[0]);
            a5 = v48;
            a4 = v51;
            v29 = *(_DWORD *)(v48 + 1744);
            v30 = *(_QWORD *)(v48 + 1728);
            if ( !v29 )
              goto LABEL_43;
            v31 = v29 - 1;
            v32 = (v29 - 1) & ((LODWORD(v54[0]) >> 9) ^ (LODWORD(v54[0]) >> 4));
            v33 = *(_QWORD *)(v30 + 8LL * v32);
            if ( v54[0] != v33 )
              break;
LABEL_31:
            v49 = v51;
            v34 = a5 + 1752;
LABEL_32:
            v52 = a5;
            sub_24044F0((__int64)v56, v34, v54);
            a5 = v52;
            a4 = v49;
LABEL_33:
            v25 = *(_QWORD *)(v25 + 8);
            if ( !v25 )
              goto LABEL_10;
          }
          v46 = 1;
          while ( v33 != -4096 )
          {
            v32 = v31 & (v46 + v32);
            v33 = *(_QWORD *)(v30 + 8LL * v32);
            if ( v54[0] == v33 )
              goto LABEL_31;
            ++v46;
          }
LABEL_43:
          v41 = *(_DWORD *)(v48 + 1776);
          v42 = *(_QWORD *)(v48 + 1760);
          if ( !v41 )
            goto LABEL_33;
          v43 = v41 - 1;
          v44 = v43 & ((LODWORD(v54[0]) >> 9) ^ (LODWORD(v54[0]) >> 4));
          v45 = *(_QWORD *)(v42 + 8LL * v44);
          if ( v54[0] != v45 )
          {
            v47 = 1;
            while ( v45 != -4096 )
            {
              v44 = v43 & (v47 + v44);
              v45 = *(_QWORD *)(v42 + 8LL * v44);
              if ( v54[0] == v45 )
                goto LABEL_45;
              ++v47;
            }
            goto LABEL_33;
          }
LABEL_45:
          v49 = v51;
          v34 = a5 + 1720;
          goto LABEL_32;
        }
      }
    }
LABEL_10:
    *(_WORD *)(v7 + 2) = sub_B52870(*(_WORD *)(v7 + 2) & 0x3F) | *(_WORD *)(v7 + 2) & 0xFFC0;
  }
LABEL_11:
  v16 = 0;
  if ( !sub_98ED60((unsigned __int8 *)v7, 0, 0, 0, 0) )
  {
    v55 = 257;
    v57 = 257;
    v20 = sub_BD2C40(72, unk_3F10A14);
    v21 = (__int64)v20;
    if ( v20 )
      sub_B549F0((__int64)v20, v7, (__int64)v56, 0, 0);
    v16 = v21;
    (*(void (__fastcall **)(unsigned int *, __int64, __int64 *, unsigned int *, unsigned int *))(*(_QWORD *)a6[11] + 16LL))(
      a6[11],
      v21,
      v54,
      a6[7],
      a6[8]);
    v22 = *a6;
    v23 = (__int64)&(*a6)[4 * *((unsigned int *)a6 + 2)];
    if ( *a6 != (unsigned int *)v23 )
    {
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v16 = *v22;
        v22 += 4;
        sub_B99FD0(v21, v16, v24);
      }
      while ( (unsigned int *)v23 != v22 );
    }
    v7 = v21;
  }
  v57 = 257;
  v17 = *a7;
  v18 = sub_AD6530(*(_QWORD *)(v7 + 8), v16);
  result = sub_B36550(a6, v17, v7, v18, (__int64)v56, 0);
  *a7 = result;
  return result;
}

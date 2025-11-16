// Function: sub_2436860
// Address: 0x2436860
//
__int64 __fastcall sub_2436860(_QWORD *a1, __int64 *a2)
{
  __int64 v2; // r15
  __int64 v5; // r13
  __int64 v7; // rax
  __int64 v8; // rdi
  unsigned __int8 *v9; // r15
  __int64 (__fastcall *v10)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  unsigned __int8 *v11; // r14
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int8 *v17; // r13
  __int64 (__fastcall *v18)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v19; // r12
  __int64 v20; // r15
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // r14
  __int64 i; // r13
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // r13
  __int64 v28; // rbx
  __int64 v29; // r13
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-100h]
  char v34[32]; // [rsp+10h] [rbp-F0h] BYREF
  __int16 v35; // [rsp+30h] [rbp-D0h]
  char v36[32]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v37; // [rsp+60h] [rbp-A0h]
  char v38[32]; // [rsp+70h] [rbp-90h] BYREF
  __int16 v39; // [rsp+90h] [rbp-70h]
  const char *v40[4]; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v41; // [rsp+C0h] [rbp-40h]

  if ( (_BYTE)qword_4FE4E28 )
    return 0;
  v2 = a1[65];
  if ( !v2 )
  {
    v5 = a1[66];
    if ( !v5 )
    {
      v32 = sub_2A3A780(a2);
      a1[66] = v32;
      v5 = v32;
    }
    v37 = 257;
    v35 = 257;
    v7 = sub_AD64C0(*(_QWORD *)(v5 + 8), 20, 0);
    v8 = a2[10];
    v9 = (unsigned __int8 *)v7;
    v10 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v8 + 24LL);
    if ( v10 == sub_920250 )
    {
      if ( *(_BYTE *)v5 > 0x15u || *v9 > 0x15u )
      {
LABEL_27:
        v41 = 257;
        v11 = (unsigned __int8 *)sub_B504D0(26, v5, (__int64)v9, (__int64)v40, 0, 0);
        (*(void (__fastcall **)(__int64, unsigned __int8 *, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
          a2[11],
          v11,
          v34,
          a2[7],
          a2[8]);
        v20 = *a2;
        v33 = *a2 + 16LL * *((unsigned int *)a2 + 2);
        if ( *a2 != v33 )
        {
          do
          {
            v21 = *(_QWORD *)(v20 + 8);
            v22 = *(_DWORD *)v20;
            v20 += 16;
            sub_B99FD0((__int64)v11, v22, v21);
          }
          while ( v33 != v20 );
        }
LABEL_12:
        v12 = a2[10];
        v13 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v12 + 16LL);
        if ( v13 == sub_9202E0 )
        {
          if ( *(_BYTE *)v5 > 0x15u || *v11 > 0x15u )
          {
LABEL_31:
            v41 = 257;
            v2 = sub_B504D0(30, v5, (__int64)v11, (__int64)v40, 0, 0);
            (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
              a2[11],
              v2,
              v36,
              a2[7],
              a2[8]);
            v23 = *a2 + 16LL * *((unsigned int *)a2 + 2);
            for ( i = *a2; v23 != i; i += 16 )
            {
              v25 = *(_QWORD *)(i + 8);
              v26 = *(_DWORD *)i;
              sub_B99FD0(v2, v26, v25);
            }
LABEL_18:
            v14 = a1[23];
            if ( v14 == 255 )
            {
LABEL_26:
              v40[0] = "hwasan.stack.base.tag";
              v41 = 259;
              sub_BD6B50((unsigned __int8 *)v2, v40);
              return v2;
            }
            v39 = 257;
            v15 = sub_AD64C0(*(_QWORD *)(v2 + 8), v14, 0);
            v16 = a2[10];
            v17 = (unsigned __int8 *)v15;
            v18 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v16 + 16LL);
            if ( v18 == sub_9202E0 )
            {
              if ( *(_BYTE *)v2 > 0x15u || *v17 > 0x15u )
                goto LABEL_34;
              if ( (unsigned __int8)sub_AC47B0(28) )
                v19 = sub_AD5570(28, v2, v17, 0, 0);
              else
                v19 = sub_AABE40(0x1Cu, (unsigned __int8 *)v2, v17);
            }
            else
            {
              v19 = v18(v16, 28u, (_BYTE *)v2, v17);
            }
            if ( v19 )
            {
LABEL_25:
              v2 = v19;
              goto LABEL_26;
            }
LABEL_34:
            v41 = 257;
            v19 = sub_B504D0(28, v2, (__int64)v17, (__int64)v40, 0, 0);
            (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
              a2[11],
              v19,
              v38,
              a2[7],
              a2[8]);
            v27 = 16LL * *((unsigned int *)a2 + 2);
            v28 = *a2;
            v29 = v28 + v27;
            while ( v29 != v28 )
            {
              v30 = *(_QWORD *)(v28 + 8);
              v31 = *(_DWORD *)v28;
              v28 += 16;
              sub_B99FD0(v19, v31, v30);
            }
            goto LABEL_25;
          }
          if ( (unsigned __int8)sub_AC47B0(30) )
            v2 = sub_AD5570(30, v5, v11, 0, 0);
          else
            v2 = sub_AABE40(0x1Eu, (unsigned __int8 *)v5, v11);
        }
        else
        {
          v2 = v13(v12, 30u, (_BYTE *)v5, v11);
        }
        if ( v2 )
          goto LABEL_18;
        goto LABEL_31;
      }
      if ( (unsigned __int8)sub_AC47B0(26) )
        v11 = (unsigned __int8 *)sub_AD5570(26, v5, v9, 0, 0);
      else
        v11 = (unsigned __int8 *)sub_AABE40(0x1Au, (unsigned __int8 *)v5, v9);
    }
    else
    {
      v11 = (unsigned __int8 *)v10(v8, 26u, (_BYTE *)v5, v9, 0);
    }
    if ( v11 )
      goto LABEL_12;
    goto LABEL_27;
  }
  return v2;
}

// Function: sub_3746040
// Address: 0x3746040
//
__int64 __fastcall sub_3746040(__int64 *a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r14d
  int v8; // eax
  unsigned int v9; // ebx
  __int64 v10; // r8
  __int64 (*v11)(); // rax
  unsigned __int32 v12; // r15d
  bool v14; // zf
  __int64 v15; // rax
  __int64 (*v16)(); // rax
  __int64 v17; // rdi
  __int64 (__fastcall *v18)(__int64, __int64, unsigned int); // rax
  int v19; // eax
  __int64 v20; // rdx
  unsigned __int64 v21; // rdx
  char v22; // al
  int v23; // edx
  __int64 (*v24)(); // rax
  unsigned int (*v25)(void); // rax
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 (__fastcall *v30)(__int64, unsigned __int16); // rax
  __int64 v31; // rsi
  __int64 v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rax
  __int64 (*v35)(); // rax
  char v36; // [rsp+Fh] [rbp-61h] BYREF
  __int64 v37; // [rsp+10h] [rbp-60h] BYREF
  __int64 v38; // [rsp+18h] [rbp-58h]
  unsigned __int64 v39; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v40; // [rsp+28h] [rbp-48h]
  char v41; // [rsp+2Ch] [rbp-44h]
  __int64 v42; // [rsp+30h] [rbp-40h]
  __int64 v43; // [rsp+38h] [rbp-38h]

  v6 = (unsigned __int16)a3;
  v8 = *a2;
  switch ( (_BYTE)v8 )
  {
    case 0x11:
      v9 = *((_DWORD *)a2 + 8);
      if ( v9 > 0x40 )
      {
        if ( v9 - (unsigned int)sub_C444A0((__int64)(a2 + 24)) > 0x40 )
          return 0;
        v11 = *(__int64 (**)())(*a1 + 88);
        v10 = **((_QWORD **)a2 + 3);
      }
      else
      {
        v10 = *((_QWORD *)a2 + 3);
        v11 = *(__int64 (**)())(*a1 + 88);
      }
      if ( v11 != sub_3740F00 )
        return ((unsigned int (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, __int64))v11)(a1, v6, v6, 11, v10);
      return 0;
    case 0x3C:
      v12 = 0;
      v25 = *(unsigned int (**)(void))(*a1 + 112);
      if ( (char *)v25 != (char *)sub_3740E90 )
        return v25();
      return v12;
    case 0x14:
      v26 = *((_QWORD *)a2 + 1);
      v27 = sub_AE4450(a1[14], v26);
      v28 = sub_AD6530(v27, v26);
      return (unsigned int)sub_3746830(a1, v28);
    case 0x12:
      v14 = !sub_AC30F0((__int64)a2);
      v15 = *a1;
      if ( v14 )
      {
        v24 = *(__int64 (**)())(v15 + 96);
        if ( v24 == sub_3740F10 )
          goto LABEL_14;
        v12 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, unsigned __int8 *))v24)(a1, v6, v6, 12, a2);
      }
      else
      {
        v16 = *(__int64 (**)())(v15 + 120);
        if ( v16 == sub_3740EA0 )
          goto LABEL_14;
        v12 = ((__int64 (__fastcall *)(__int64 *, unsigned __int8 *))v16)(a1, a2);
      }
      if ( v12 )
        return v12;
LABEL_14:
      v17 = a1[16];
      v18 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v17 + 32LL);
      if ( v18 == sub_2D42F30 )
      {
        v19 = sub_AE2980(a1[14], 0)[1];
        switch ( v19 )
        {
          case 1:
            v37 = 2;
            v19 = 2;
            v38 = 0;
            break;
          case 2:
            v37 = 3;
            v19 = 3;
            v38 = 0;
            break;
          case 4:
            v37 = 4;
            v38 = 0;
            break;
          case 8:
            v37 = 5;
            v19 = 5;
            v38 = 0;
            break;
          case 16:
            v37 = 6;
            v19 = 6;
            v38 = 0;
            break;
          case 32:
            v37 = 7;
            v19 = 7;
            v38 = 0;
            break;
          case 64:
            v37 = 8;
            v19 = 8;
            v38 = 0;
            break;
          case 128:
            v37 = 9;
            v19 = 9;
            v38 = 0;
            break;
          default:
            v37 = 0;
            v38 = 0;
LABEL_24:
            v42 = sub_3007260((__int64)&v37);
            v43 = v20;
            v21 = v42;
            v22 = v43;
LABEL_25:
            v39 = v21;
            LOBYTE(v40) = v22;
            v40 = sub_CA1930(&v39);
            if ( v40 > 0x40 )
              sub_C43690((__int64)&v39, 0, 0);
            else
              v39 = 0;
            v41 = 0;
            v12 = 0;
            sub_C41980((void **)a2 + 3, (__int64)&v39, 0, &v36);
            if ( v36 )
            {
              v33 = (__int64 *)sub_BD5C60((__int64)a2);
              v34 = sub_ACCFD0(v33, (__int64)&v39);
              v12 = sub_3746830(a1, v34);
              if ( v12 )
              {
                v35 = *(__int64 (**)())(*a1 + 64);
                if ( v35 == sub_3740EE0 )
                  v12 = 0;
                else
                  v12 = ((__int64 (__fastcall *)(__int64 *, _QWORD, _QWORD, __int64, _QWORD))v35)(
                          a1,
                          (unsigned __int16)v37,
                          v6,
                          220,
                          v12);
              }
            }
            if ( v40 > 0x40 )
            {
              if ( v39 )
                j_j___libc_free_0_0(v39);
            }
            return v12;
        }
      }
      else
      {
        LOWORD(v19) = v18(v17, a1[14], 0);
        v38 = 0;
        LOWORD(v37) = v19;
        if ( !(_WORD)v19 )
          goto LABEL_24;
        v19 = (unsigned __int16)v19;
        if ( (_WORD)v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
          BUG();
      }
      v32 = 16LL * (v19 - 1);
      v21 = *(_QWORD *)&byte_444C4A0[v32];
      v22 = byte_444C4A0[v32 + 8];
      goto LABEL_25;
    case 5:
      v23 = *((unsigned __int16 *)a2 + 1);
      break;
    default:
      if ( (unsigned __int8)v8 <= 0x1Cu )
      {
        v12 = 0;
        if ( (unsigned __int8)(v8 - 12) <= 1u )
        {
          v29 = a1[16];
          v30 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v29 + 552LL);
          if ( v30 == sub_2EC09E0 )
            v31 = *(_QWORD *)(v29 + 8LL * (unsigned __int16)a3 + 112);
          else
            v31 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v30)(v29, (unsigned __int16)a3, 0);
          v12 = sub_3741980((__int64)a1, v31, a3, a4, a5, a6);
          sub_2F26260(
            *(_QWORD *)(a1[5] + 744),
            *(__int64 **)(a1[5] + 752),
            a1 + 10,
            *(_QWORD *)(a1[15] + 8) - 400LL,
            v12);
        }
        return v12;
      }
      v23 = v8 - 29;
      break;
  }
  if ( sub_3745920(a1, (__int64)a2, v23)
    || *a2 > 0x1Cu && (*(unsigned __int8 (__fastcall **)(__int64 *, unsigned __int8 *))(*a1 + 24))(a1, a2) )
  {
    return (unsigned int)sub_3742170((__int64)a1, (__int64)a2);
  }
  return 0;
}

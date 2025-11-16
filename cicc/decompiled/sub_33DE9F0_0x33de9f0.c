// Function: sub_33DE9F0
// Address: 0x33de9f0
//
__int64 __fastcall sub_33DE9F0(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v4; // r13
  unsigned int v6; // ebx
  char v7; // cl
  unsigned __int8 v8; // r8
  int v9; // eax
  unsigned int v10; // eax
  char v11; // r15
  int v12; // ebx
  __int64 *v13; // rax
  unsigned int v14; // r13d
  __int64 *v15; // rax
  __int64 v16; // rax
  __int64 *v17; // rax
  unsigned int v18; // r8d
  unsigned int v19; // eax
  __int64 v20; // rdx
  __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 *v24; // rdi
  int v25; // edx
  unsigned int v26; // r13d
  bool v27; // zf
  __int64 v28; // rax
  unsigned int v29; // r15d
  unsigned int v30; // ebx
  int v31; // eax
  int v32; // eax
  int v33; // eax
  int v34; // eax
  unsigned __int64 v35; // rax
  int v36; // ebx
  int v37; // [rsp+4h] [rbp-BCh]
  __int64 v38; // [rsp+8h] [rbp-B8h]
  char v39; // [rsp+10h] [rbp-B0h]
  __int64 v40; // [rsp+10h] [rbp-B0h]
  __int64 v42; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v43; // [rsp+28h] [rbp-98h]
  __int64 v44; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v45; // [rsp+38h] [rbp-88h]
  __int64 v46[2]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v47; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v48; // [rsp+58h] [rbp-68h]
  __int64 (__fastcall *v49)(_QWORD *, __int64, int); // [rsp+60h] [rbp-60h] BYREF
  __int64 (__fastcall *v50)(__int64, __int64 *); // [rsp+68h] [rbp-58h]
  _QWORD *v51; // [rsp+70h] [rbp-50h] BYREF
  unsigned int v52; // [rsp+78h] [rbp-48h]
  void (__fastcall *v53)(_QWORD **, _QWORD **, __int64); // [rsp+80h] [rbp-40h] BYREF
  __int64 (__fastcall *v54)(__int64, __int64 *); // [rsp+88h] [rbp-38h]

  while ( 2 )
  {
    if ( a4 > 5 )
    {
LABEL_2:
      LODWORD(v4) = 0;
      return (unsigned int)v4;
    }
    v6 = a4;
    v50 = sub_33C9090;
    v4 = a3;
    v49 = sub_33C7E30;
    v53 = 0;
    sub_33C7E30(&v51, (__int64)&v47, 2);
    v54 = v50;
    v53 = (void (__fastcall *)(_QWORD **, _QWORD **, __int64))v49;
    v39 = sub_33CA8D0((_QWORD *)a2, v4, (__int64)&v51, v7, v8);
    if ( v53 )
      v53(&v51, &v51, 3);
    if ( v49 )
      v49(&v47, (__int64)&v47, 3);
    if ( v39 )
      goto LABEL_24;
    v9 = *(_DWORD *)(a2 + 24);
    if ( v9 <= 214 )
    {
      if ( v9 <= 55 )
        goto LABEL_15;
      switch ( v9 )
      {
        case 56:
          if ( (*(_BYTE *)(a2 + 28) & 1) == 0 )
            goto LABEL_15;
          v29 = v6 + 1;
          if ( !(unsigned __int8)sub_33DE9F0(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                   v6 + 1) )
            goto LABEL_49;
          goto LABEL_24;
        case 57:
          v26 = v6 + 1;
          v27 = !sub_33CF170(**(_QWORD **)(a2 + 40));
          v28 = *(_QWORD *)(a2 + 40);
          if ( v27 )
          {
            sub_33DD090((__int64)&v51, a1, *(_QWORD *)(v28 + 40), *(_QWORD *)(v28 + 48), v26);
            sub_33DD090((__int64)&v47, a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v26);
            v36 = sub_C771C0(&v47, (__int64)&v51);
            sub_969240((__int64 *)&v49);
            sub_969240(&v47);
            sub_969240((__int64 *)&v53);
            sub_969240((__int64 *)&v51);
            LODWORD(v4) = BYTE1(v36);
            if ( BYTE1(v36) )
              LODWORD(v4) = v36;
            return (unsigned int)v4;
          }
          a2 = *(_QWORD *)(v28 + 40);
          a3 = *(_QWORD *)(v28 + 48);
          a4 = v6 + 1;
          continue;
        case 58:
          v33 = *(_DWORD *)(a2 + 28);
          if ( (v33 & 2) == 0 && (v33 & 1) == 0 )
            goto LABEL_15;
          v29 = v6 + 1;
          if ( !(unsigned __int8)sub_33DE9F0(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                   v6 + 1) )
            goto LABEL_15;
LABEL_49:
          if ( (unsigned __int8)sub_33DE9F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v29) )
            goto LABEL_24;
          goto LABEL_15;
        case 59:
        case 60:
          if ( (*(_BYTE *)(a2 + 28) & 4) == 0 )
            goto LABEL_15;
          goto LABEL_26;
        case 83:
        case 183:
        case 187:
          v14 = v6 + 1;
          if ( (unsigned __int8)sub_33DE9F0(
                                  a1,
                                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                  *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                  v6 + 1) )
            goto LABEL_24;
          goto LABEL_29;
        case 180:
          v30 = v6 + 1;
          sub_33DD090(
            (__int64)&v47,
            a1,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
            v30);
          LOBYTE(v31) = sub_986C60((__int64 *)&v49, (_DWORD)v50 - 1);
          LODWORD(v4) = v31;
          if ( (_BYTE)v31 )
            goto LABEL_59;
          sub_33DD090((__int64)&v51, a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v30);
          LOBYTE(v32) = sub_986C60((__int64 *)&v53, (_DWORD)v54 - 1);
          LODWORD(v4) = v32;
          if ( !(_BYTE)v32 )
            goto LABEL_53;
          goto LABEL_57;
        case 181:
          v30 = v6 + 1;
          sub_33DD090(
            (__int64)&v47,
            a1,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
            v30);
          LOBYTE(v34) = sub_986C60(&v47, v48 - 1);
          LODWORD(v4) = v34;
          if ( (_BYTE)v34 && !sub_9867B0((__int64)&v49) )
            goto LABEL_59;
          sub_33DD090((__int64)&v51, a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v30);
          if ( v52 > 0x40 )
            v35 = v51[(v52 - 1) >> 6];
          else
            v35 = (unsigned __int64)v51;
          if ( (v35 & (1LL << ((unsigned __int8)v52 - 1))) == 0 )
            goto LABEL_53;
          if ( (unsigned int)v54 <= 0x40 )
          {
            if ( v53 )
            {
              LODWORD(v4) = 1;
              sub_969240((__int64 *)&v53);
LABEL_58:
              sub_969240((__int64 *)&v51);
LABEL_59:
              sub_969240((__int64 *)&v49);
              sub_969240(&v47);
              return (unsigned int)v4;
            }
          }
          else
          {
            v37 = (int)v54;
            LODWORD(v4) = 1;
            if ( v37 != (unsigned int)sub_C444A0((__int64)&v53) )
            {
LABEL_57:
              sub_969240((__int64 *)&v53);
              goto LABEL_58;
            }
          }
LABEL_53:
          if ( sub_9867B0((__int64)&v49) || (LODWORD(v4) = 1, sub_9867B0((__int64)&v53)) )
          {
            LODWORD(v4) = sub_33DE9F0(
                            a1,
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                            v30);
            if ( (_BYTE)v4 )
              LODWORD(v4) = sub_33DE9F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), v30);
          }
          goto LABEL_57;
        case 182:
          v14 = v6 + 1;
          if ( !(unsigned __int8)sub_33DE9F0(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                   v6 + 1) )
            goto LABEL_2;
LABEL_29:
          v15 = *(__int64 **)(a2 + 40);
          a4 = v14;
          a2 = *v15;
          a3 = v15[1];
          continue;
        case 189:
        case 193:
        case 194:
        case 197:
        case 200:
        case 201:
        case 213:
        case 214:
LABEL_26:
          v13 = *(__int64 **)(a2 + 40);
          a4 = v6 + 1;
          a2 = *v13;
          a3 = v13[1];
          continue;
        case 190:
          v25 = *(_DWORD *)(a2 + 28);
          v17 = *(__int64 **)(a2 + 40);
          v18 = v6 + 1;
          if ( (v25 & 2) != 0 || (v25 & 1) != 0 )
            goto LABEL_44;
          sub_33DD090((__int64)&v47, a1, *v17, v17[1], v18);
          if ( sub_986C60((__int64 *)&v49, 0) )
            goto LABEL_23;
          sub_33DD090(
            (__int64)&v51,
            a1,
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
            v6 + 1);
          sub_D95160((__int64)&v44, (__int64)&v51);
          sub_969240((__int64 *)&v53);
          sub_969240((__int64 *)&v51);
          if ( !sub_986EE0((__int64)&v44, v48) )
            goto LABEL_82;
          sub_9865C0((__int64)&v51, (__int64)&v49);
          sub_C47AC0((__int64)&v51, (__int64)&v44);
          if ( !sub_9867B0((__int64)&v51) )
          {
            sub_969240((__int64 *)&v51);
            sub_969240(&v44);
            goto LABEL_23;
          }
          sub_969240((__int64 *)&v51);
LABEL_82:
          v24 = &v44;
          goto LABEL_41;
        case 191:
        case 192:
          v17 = *(__int64 **)(a2 + 40);
          v18 = v6 + 1;
          if ( (*(_BYTE *)(a2 + 28) & 4) == 0 )
          {
            sub_33DD090((__int64)&v47, a1, *v17, v17[1], v18);
            if ( !sub_986C60((__int64 *)&v49, (_DWORD)v50 - 1) )
            {
              sub_33DD090(
                (__int64)&v51,
                a1,
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                v6 + 1);
              v19 = v52;
              v45 = v52;
              if ( v52 <= 0x40 )
              {
                v20 = (__int64)v51;
                goto LABEL_36;
              }
              sub_C43780((__int64)&v44, (const void **)&v51);
              v19 = v45;
              if ( v45 <= 0x40 )
              {
                v20 = v44;
LABEL_36:
                v21 = ~v20;
                v22 = 0;
                if ( v19 )
                  v22 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v19;
                v23 = v22 & v21;
              }
              else
              {
                sub_C43D10((__int64)&v44);
                v19 = v45;
                v23 = v44;
              }
              v43 = v19;
              v42 = v23;
              sub_969240((__int64 *)&v53);
              sub_969240((__int64 *)&v51);
              if ( !sub_986EE0((__int64)&v42, v48) )
              {
LABEL_40:
                v24 = &v42;
LABEL_41:
                sub_969240(v24);
                sub_969240((__int64 *)&v49);
                sub_969240(&v47);
LABEL_15:
                sub_33DD090((__int64)&v51, a1, a2, v4, v6);
                v12 = (int)v54;
                if ( (unsigned int)v54 <= 0x40 )
                {
                  LOBYTE(v4) = v53 != 0;
                }
                else
                {
                  LOBYTE(v4) = v12 != (unsigned int)sub_C444A0((__int64)&v53);
                  if ( v53 )
                    j_j___libc_free_0_0((unsigned __int64)v53);
                }
                if ( v52 > 0x40 && v51 )
                  j_j___libc_free_0_0((unsigned __int64)v51);
                return (unsigned int)v4;
              }
              sub_9865C0((__int64)&v51, (__int64)&v49);
              sub_C48380((__int64)&v51, (__int64)&v42);
              if ( sub_9867B0((__int64)&v51) )
              {
                sub_969240((__int64 *)&v51);
                goto LABEL_40;
              }
              sub_969240((__int64 *)&v51);
              sub_969240(&v42);
            }
LABEL_23:
            sub_969240((__int64 *)&v49);
            sub_969240(&v47);
LABEL_24:
            LODWORD(v4) = 1;
            return (unsigned int)v4;
          }
LABEL_44:
          a2 = *v17;
          a3 = v17[1];
          a4 = v18;
          continue;
        case 205:
        case 206:
          if ( !(unsigned __int8)sub_33DE9F0(
                                   a1,
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                                   *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                                   v6 + 1) )
            goto LABEL_2;
          v16 = *(_QWORD *)(a2 + 40);
          a4 = v6 + 1;
          a2 = *(_QWORD *)(v16 + 80);
          a3 = *(_QWORD *)(v16 + 88);
          continue;
        default:
          goto LABEL_15;
      }
    }
    break;
  }
  if ( v9 != 373 )
    goto LABEL_15;
  v38 = **(_QWORD **)(a1 + 40);
  v40 = *(_QWORD *)(**(_QWORD **)(a2 + 40) + 96LL) + 24LL;
  v10 = sub_33C9580(a2, v4);
  sub_988CD0((__int64)&v47, v38, v10);
  sub_9865C0((__int64)&v42, v40);
  sub_AADBC0((__int64)&v51, &v42);
  sub_AB5480((__int64)&v44, (__int64)&v47, (__int64 *)&v51);
  sub_969240((__int64 *)&v53);
  sub_969240((__int64 *)&v51);
  sub_969240(&v42);
  sub_969240((__int64 *)&v49);
  sub_969240(&v47);
  sub_9691E0((__int64)&v51, v45, 0, 0, 0);
  v11 = sub_AB1B10((__int64)&v44, (__int64)&v51);
  sub_969240((__int64 *)&v51);
  if ( v11 )
  {
    sub_969240(v46);
    sub_969240(&v44);
    goto LABEL_15;
  }
  LODWORD(v4) = 1;
  sub_969240(v46);
  sub_969240(&v44);
  return (unsigned int)v4;
}

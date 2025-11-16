// Function: sub_18FDB50
// Address: 0x18fdb50
//
unsigned __int64 __fastcall sub_18FDB50(__int64 *a1, __int64 *a2)
{
  __int64 v3; // rdx
  __int64 *v4; // rbx
  __int64 *v5; // rsi
  __int64 v6; // rax
  unsigned __int64 v8; // r15
  unsigned __int64 v9; // r9
  unsigned __int64 v10; // r13
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // r11
  unsigned __int64 v13; // r12
  char *v14; // r8
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  __int64 v18; // r12
  __int64 v19; // r11
  __int64 v20; // r11
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  char *v24; // rax
  char *v25; // rsi
  unsigned __int64 v26; // r12
  unsigned __int64 v27; // r15
  unsigned __int64 v28; // r15
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v32; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v33; // [rsp+20h] [rbp-E0h]
  char *v34; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v35; // [rsp+38h] [rbp-C8h]
  unsigned __int64 v36; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v37[8]; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v38; // [rsp+90h] [rbp-70h] BYREF
  __int64 v39; // [rsp+98h] [rbp-68h] BYREF
  __int64 v40; // [rsp+A0h] [rbp-60h]
  __int64 v41; // [rsp+A8h] [rbp-58h]
  __int64 v42; // [rsp+B0h] [rbp-50h]
  __int64 v43; // [rsp+B8h] [rbp-48h]
  __int64 v44; // [rsp+C0h] [rbp-40h]
  __int64 v45; // [rsp+C8h] [rbp-38h]
  char v46[8]; // [rsp+D0h] [rbp-30h] BYREF
  _BYTE v47[40]; // [rsp+D8h] [rbp-28h] BYREF

  if ( !byte_4F99930[0] && (unsigned int)sub_2207590(byte_4F99930) )
  {
    v30 = unk_4FA04C8;
    if ( !unk_4FA04C8 )
      v30 = 0xFF51AFD7ED558CCDLL;
    qword_4F99938 = v30;
    sub_2207640(byte_4F99930);
  }
  v3 = qword_4F99938;
  if ( a2 == a1 )
    return sub_1593600(&v38, 0, qword_4F99938);
  v4 = a1 + 3;
  v5 = &v39;
  v38 = *a1;
  if ( a1 + 3 == a2 )
    return sub_1593600(&v38, (char *)v5 - (char *)&v38, v3);
  while ( ++v5 != (__int64 *)v47 )
  {
    v6 = *v4;
    v4 += 3;
    *(v5 - 1) = v6;
    if ( v4 == a2 )
      return sub_1593600(&v38, (char *)v5 - (char *)&v38, v3);
  }
  sub_15938B0(v37, &v38, v3);
  v8 = v37[6];
  v35 = 64;
  v9 = v37[0];
  v10 = v37[1];
  v36 = v37[2];
  v11 = v37[3];
  v12 = v37[4];
  v13 = v37[5];
  v14 = v47;
  while ( 1 )
  {
    v23 = *v4;
    v24 = (char *)&v39;
    while ( 1 )
    {
      v4 += 3;
      *((_QWORD *)v24 - 1) = v23;
      v25 = v24;
      if ( a2 == v4 )
        break;
      v24 += 8;
      if ( v24 == v14 )
        break;
      v23 = *v4;
    }
    v31 = v9;
    v32 = v11;
    v33 = v12;
    v34 = v14;
    sub_18FBDC0((char *)&v38, v25, v46);
    v15 = v10 + v32;
    v14 = v34;
    v16 = 0xB492B66FBE98F273LL * v33 + v38;
    v10 = v43 + v32 - 0x4B6D499041670D8DLL * __ROL8__(v44 + v33 + v10, 22);
    v9 = 0xB492B66FBE98F273LL * __ROL8__(v13 + v36, 31);
    v17 = v8 ^ (0xB492B66FBE98F273LL * __ROL8__(v31 + v39 + v15, 27));
    v18 = __ROR8__(v17 + v16 + v13 + v41, 21);
    v19 = v16 + v40 + v39;
    v11 = v41 + v19;
    v20 = v16 + __ROL8__(v19, 20);
    v21 = v42 + v8 + v9;
    v12 = v18 + v20;
    v35 += v25 - (char *)&v38;
    v22 = v21 + v43 + v44;
    v13 = v45 + v22;
    v8 = __ROL8__(v22, 20) + v21 + __ROR8__(v10 + v21 + v40 + v45, 21);
    if ( a2 == v4 )
      break;
    v36 = v17;
  }
  v26 = 0x9DDFEA08EB382D69LL
      * (((0x9DDFEA08EB382D69LL * (v13 ^ v11)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v13 ^ v11)) ^ v13);
  v27 = 0x9DDFEA08EB382D69LL * (((0x9DDFEA08EB382D69LL * (v8 ^ v12)) >> 47) ^ (0x9DDFEA08EB382D69LL * (v8 ^ v12)) ^ v8);
  v28 = 0xB492B66FBE98F273LL * (v35 ^ (v35 >> 47)) + v9 - 0x622015F714C7D297LL * ((v27 >> 47) ^ v27);
  v29 = 0x9DDFEA08EB382D69LL
      * (v28 ^ (0xB492B66FBE98F273LL * ((v10 >> 47) ^ v10) + v17 - 0x622015F714C7D297LL * ((v26 >> 47) ^ v26)));
  return 0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL * (v29 ^ v28 ^ (v29 >> 47))) >> 47) ^ (0x9DDFEA08EB382D69LL * (v29 ^ v28 ^ (v29 >> 47))));
}

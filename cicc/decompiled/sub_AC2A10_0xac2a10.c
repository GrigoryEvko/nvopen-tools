// Function: sub_AC2A10
// Address: 0xac2a10
//
unsigned __int64 __fastcall sub_AC2A10(unsigned __int64 *a1, _QWORD *a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // r8
  unsigned __int64 v4; // rcx
  unsigned __int64 v5; // r10
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // rax
  unsigned __int64 result; // rax
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r8
  unsigned __int64 v14; // rcx
  __int64 v15; // r11
  unsigned __int64 v16; // r10
  unsigned __int64 v17; // rbx
  unsigned __int64 v18; // r10
  unsigned __int64 v19; // rcx
  unsigned __int64 v20; // rcx
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  __int64 v23; // r10
  unsigned __int64 v24; // rdx
  __int64 v25; // r9

  v2 = a1[1];
  v3 = a1[3];
  v4 = a1[4];
  v5 = a1[5];
  v6 = a2[1] + v2 + *a1;
  v7 = v4 + v2;
  v4 *= 0xB492B66FBE98F273LL;
  v8 = 0xB492B66FBE98F273LL * __ROL8__(v3 + v6, 27);
  *a1 = v8;
  result = a1[6] ^ v8;
  v10 = 0xB492B66FBE98F273LL * __ROL8__(a2[6] + v7, 22);
  *a1 = result;
  a1[1] = v10;
  v11 = a2[5] + v3 + v10;
  v12 = a1[2];
  a1[3] = v4;
  a1[4] = result + v5;
  a1[1] = v11;
  v13 = 0xB492B66FBE98F273LL * __ROL8__(v5 + v12, 31);
  a1[2] = v13;
  v14 = *a2 + v4;
  a1[3] = v14;
  v15 = a2[3];
  v16 = __ROR8__(result + v5 + v14 + v15, 21);
  a1[4] = v16;
  v17 = v14 + a2[1] + a2[2];
  a1[3] = v15 + v17;
  v18 = __ROL8__(v17, 20) + v14 + v16;
  v19 = a1[6];
  a1[4] = v18;
  v20 = v13 + v19;
  a1[5] = v20;
  v21 = a2[2] + v11;
  a1[6] = v21;
  v22 = a2[4] + v20;
  a1[5] = v22;
  v23 = a2[7];
  v24 = __ROR8__(v22 + v23 + v21, 21);
  a1[6] = v24;
  v25 = a2[5] + a2[6];
  a1[2] = result;
  *a1 = v13;
  a1[5] = v23 + v22 + v25;
  a1[6] = __ROL8__(v22 + v25, 20) + v22 + v24;
  return result;
}

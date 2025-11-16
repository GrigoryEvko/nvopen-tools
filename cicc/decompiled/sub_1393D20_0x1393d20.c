// Function: sub_1393D20
// Address: 0x1393d20
//
void __fastcall sub_1393D20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r13
  unsigned __int8 v6; // al
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // rsi
  char v12; // cl
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // rcx
  __int64 v25; // rdi
  __int64 v26; // r13
  __int64 v27; // rsi
  __int64 v28; // r13
  __int64 v29; // rsi
  __int64 v30; // r10
  __int64 *v31; // r9
  __int64 *v32; // r13
  __int64 *v33; // r14
  __int64 v34; // rbx
  unsigned __int8 v35; // al
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rbx
  unsigned __int8 v40; // al
  __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rdx
  __int64 v45; // r12
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r12
  __int64 v54; // rax
  unsigned __int64 v55; // [rsp+8h] [rbp-58h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+20h] [rbp-40h]
  __int64 v60; // [rsp+28h] [rbp-38h]

  v4 = a2;
  switch ( *(_BYTE *)(a2 + 16) )
  {
    case 0x18:
    case 0x1A:
    case 0x1B:
    case 0x1C:
    case 0x1E:
    case 0x1F:
    case 0x20:
    case 0x21:
    case 0x22:
    case 0x39:
    case 0x49:
    case 0x4A:
    case 0x4B:
    case 0x4C:
    case 0x50:
    case 0x51:
    case 0x52:
    case 0x58:
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
        goto LABEL_14;
      return;
    case 0x19:
      if ( (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) != 0 )
      {
        v39 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
        if ( v39 )
        {
          if ( *(_BYTE *)(*(_QWORD *)v39 + 8LL) == 15 )
          {
            v40 = *(_BYTE *)(v39 + 16);
            if ( v40 > 3u )
            {
              if ( v40 == 5 )
              {
                if ( (unsigned int)*(unsigned __int16 *)(v39 + 18) - 51 > 1
                  && (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), v39, 0, 0) )
                {
                  sub_1391610(a1, v39, v52);
                }
              }
              else
              {
                sub_13848E0(*(_QWORD *)(a1 + 24), v39, 0, 0);
              }
            }
            else
            {
              v41 = *(_QWORD *)(a1 + 24);
              v42 = sub_14C81A0(v39);
              v43 = v41;
              if ( (unsigned __int8)sub_13848E0(v41, v39, 0, v42) )
              {
                v53 = *(_QWORD *)(a1 + 24);
                v54 = sub_14C8160(v43, v39, v44);
                sub_13848E0(v53, v39, 1u, v54);
              }
            }
            v45 = *(_QWORD *)(a1 + 32);
            v46 = *(unsigned int *)(v45 + 8);
            if ( (unsigned int)v46 >= *(_DWORD *)(v45 + 12) )
            {
              sub_16CD150(*(_QWORD *)(a1 + 32), v45 + 16, 0, 8);
              v46 = *(unsigned int *)(v45 + 8);
            }
            *(_QWORD *)(*(_QWORD *)v45 + 8 * v46) = v39;
            ++*(_DWORD *)(v45 + 8);
          }
        }
      }
      return;
    case 0x1D:
      sub_1393380(a1, a2 & 0xFFFFFFFFFFFFFFFBLL);
      return;
    case 0x23:
    case 0x24:
    case 0x25:
    case 0x26:
    case 0x27:
    case 0x28:
    case 0x29:
    case 0x2A:
    case 0x2B:
    case 0x2C:
    case 0x2D:
    case 0x2E:
    case 0x2F:
    case 0x30:
    case 0x31:
    case 0x32:
    case 0x33:
    case 0x34:
      sub_1392050(a1, a2);
      return;
    case 0x35:
      v25 = *(_QWORD *)(a1 + 24);
      v24 = 0;
      goto LABEL_37;
    case 0x36:
    case 0x56:
      v11 = *(_QWORD *)(a2 - 24);
      v12 = 1;
      v13 = v4;
      goto LABEL_12;
    case 0x37:
      v13 = *(_QWORD *)(a2 - 24);
      v12 = 0;
      v11 = *(_QWORD *)(a2 - 48);
      goto LABEL_12;
    case 0x38:
      sub_1392250(a1, a2);
      return;
    case 0x3A:
      v13 = *(_QWORD *)(a2 - 72);
      v12 = 0;
      v11 = *(_QWORD *)(a2 - 24);
      goto LABEL_12;
    case 0x3B:
      v13 = *(_QWORD *)(a2 - 48);
      v12 = 0;
      v11 = *(_QWORD *)(a2 - 24);
      goto LABEL_12;
    case 0x3C:
    case 0x3D:
    case 0x3E:
    case 0x3F:
    case 0x40:
    case 0x41:
    case 0x42:
    case 0x43:
    case 0x44:
    case 0x47:
    case 0x48:
      v5 = *(_QWORD *)(a2 - 24);
      if ( *(_BYTE *)(*(_QWORD *)v5 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
      {
        v6 = *(_BYTE *)(v5 + 16);
        if ( v6 > 3u )
        {
          if ( v6 == 5 )
          {
            if ( (unsigned int)*(unsigned __int16 *)(v5 + 18) - 51 > 1
              && (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 24), 0, 0) )
            {
              sub_1391610(a1, v5, v47);
            }
          }
          else
          {
            sub_13848E0(*(_QWORD *)(a1 + 24), *(_QWORD *)(a2 - 24), 0, 0);
          }
        }
        else
        {
          v7 = *(_QWORD *)(a1 + 24);
          v8 = sub_14C81A0(*(_QWORD *)(a2 - 24));
          v9 = v7;
          if ( (unsigned __int8)sub_13848E0(v7, v5, 0, v8) )
          {
            v49 = *(_QWORD *)(a1 + 24);
            v50 = sub_14C8160(v9, v5, v10);
            sub_13848E0(v49, v5, 1u, v50);
          }
        }
        if ( v5 != a2 )
          sub_1391C50(a1, v5, a2, 0);
      }
      return;
    case 0x45:
      v4 = *(_QWORD *)(a2 - 24);
      v14 = sub_14C8190();
      v15 = *(_BYTE *)(v4 + 16);
      if ( v15 > 3u )
        goto LABEL_39;
      goto LABEL_15;
    case 0x46:
LABEL_14:
      v14 = sub_14C8160(a1, a2, a3);
      v15 = *(_BYTE *)(a2 + 16);
      if ( v15 <= 3u )
      {
LABEL_15:
        v16 = *(_QWORD *)(a1 + 24);
        v17 = sub_14C81A0(v4);
        v18 = v16;
        if ( !(unsigned __int8)sub_13848E0(v16, v4, 0, v17) )
          return;
        v20 = *(_QWORD *)(a1 + 24);
        v21 = sub_14C8160(v18, v4, v19);
        v22 = v4;
        v23 = 1;
        v24 = v21;
        v25 = v20;
        goto LABEL_17;
      }
LABEL_39:
      if ( v15 != 5 )
      {
        v25 = *(_QWORD *)(a1 + 24);
        v24 = v14;
LABEL_37:
        v22 = v4;
        v23 = 0;
LABEL_17:
        sub_13848E0(v25, v22, v23, v24);
        return;
      }
      if ( (unsigned int)*(unsigned __int16 *)(v4 + 18) - 51 > 1 )
      {
        if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), v4, 0, 0) )
          sub_1391610(a1, v4, v38);
      }
      return;
    case 0x4D:
      v30 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
      {
        v31 = *(__int64 **)(a2 - 8);
        v32 = &v31[v30];
      }
      else
      {
        v32 = (__int64 *)a2;
        v31 = (__int64 *)(a2 - v30 * 8);
      }
      if ( v31 != v32 )
      {
        v33 = v31;
        do
        {
          v34 = *v33;
          if ( *(_BYTE *)(*(_QWORD *)*v33 + 8LL) == 15 && *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
          {
            v35 = *(_BYTE *)(v34 + 16);
            if ( v35 > 3u )
            {
              if ( v35 == 5 )
              {
                if ( (unsigned int)*(unsigned __int16 *)(v34 + 18) - 51 > 1 )
                {
                  v56 &= 0xFFFFFFFF00000000LL;
                  if ( (unsigned __int8)sub_13848E0(*(_QWORD *)(a1 + 24), v34, v56, 0) )
                    sub_1391610(a1, v34, v48);
                }
              }
              else
              {
                v59 &= 0xFFFFFFFF00000000LL;
                sub_13848E0(*(_QWORD *)(a1 + 24), v34, v59, 0);
              }
            }
            else
            {
              v57 = *(_QWORD *)(a1 + 24);
              v36 = sub_14C81A0(*v33);
              v60 &= 0xFFFFFFFF00000000LL;
              if ( (unsigned __int8)sub_13848E0(v57, v34, v60, v36) )
              {
                v58 = *(_QWORD *)(a1 + 24);
                v51 = sub_14C8160(v58, v34, v37);
                v55 = v55 & 0xFFFFFFFF00000000LL | 1;
                sub_13848E0(v58, v34, 1u, v51);
              }
            }
            if ( a2 != v34 )
              sub_1391C50(a1, v34, a2, 0);
          }
          v33 += 3;
        }
        while ( v32 != v33 );
      }
      return;
    case 0x4E:
      sub_1393CF0(a1, a2);
      return;
    case 0x4F:
      v26 = *(_QWORD *)(a2 - 24);
      v27 = *(_QWORD *)(a2 - 48);
      goto LABEL_19;
    case 0x53:
      v11 = *(_QWORD *)(a2 - 48);
      v12 = 1;
      v13 = v4;
      goto LABEL_12;
    case 0x54:
      v28 = *(_QWORD *)(a2 - 48);
      v29 = *(_QWORD *)(a2 - 72);
      goto LABEL_21;
    case 0x55:
      v26 = *(_QWORD *)(a2 - 48);
      v27 = *(_QWORD *)(a2 - 72);
LABEL_19:
      sub_1391F40(a1, v27, v4, 0);
      sub_1391F40(a1, v26, v4, 0);
      return;
    case 0x57:
      v28 = *(_QWORD *)(a2 - 24);
      v29 = *(_QWORD *)(a2 - 48);
LABEL_21:
      sub_1391F40(a1, v29, v4, 0);
      v12 = 0;
      v13 = v4;
      v11 = v28;
LABEL_12:
      sub_13911E0(a1, v11, v13, v12);
      return;
  }
}
